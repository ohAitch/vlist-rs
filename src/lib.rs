#![feature(const_panic)]
#![feature(const_in_array_repeat_expressions)]
#![feature(array_map)]
#![feature(untagged_unions,min_const_generics)]
#![feature(arbitrary_enum_discriminant)]

#![allow(unused, dead_code)] // TODO

use std::convert::{TryInto,TryFrom};
use std::mem;

use core::fmt;
use core::fmt::{Debug};
use core::ops::{IndexMut};

#[macro_use]
mod sgbr {
    #![allow(unused_macros)]
    pub struct Sgbr<F: Fn()>(pub F);
    impl<F: Fn()> Drop for Sgbr<F> { fn drop(&mut self){
        if std::thread::panicking() { self.0() }
    } }
    
    macro_rules! sgbr {
        ($($data:tt)*) => { let _raii = $crate::sgbr::Sgbr(|| {eprintln!($($data)*);});};
    }
}

//TODO 4096 maybe
const PAGE_SIZE: usize = 64;
const PAGES: usize = 1024;
type Elem = u32;

#[derive(Clone, Copy)]
struct Index(u16,u8,u8);
impl Index {
    fn of(a:usize, b:usize, c:usize)-> Self { Self(a.try_into().unwrap(), b.try_into().unwrap(), c.try_into().unwrap())}
    fn of32(a:u32, b:u32, c:u32)-> Self { Self(a.try_into().unwrap(), b.try_into().unwrap(), c.try_into().unwrap())}
}
impl std::ops::Add<u8> for Index { 
    type Output = Self; fn add(self, ofs: u8)-> Self { let mut s = self; s.2 += ofs; s }
}
impl std::ops::Sub<u8> for Index {
    type Output = Self; fn sub(self, ofs: u8)-> Self { let mut s = self; s.2 -= ofs; s }
}
impl Debug for Index {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f,"{}-{}-{}", self.0, self.1, self.2)
    }
}
impl Into<Elem> for Index { fn into(self) -> Elem { (self.0 as u32)*0x1_0000 ^ (self.1 as u32)*0x100 ^ (self.2 as u32) } }
impl From<Elem> for Index { fn from(e: Elem) -> Self { Self::of32(e>>16, e>>8 & 0xff, e&0xff) } }

//NOTE not strictly speaking cloneable, ideally this would be a sealed bit-clone
// trait that was only used by the allocator
#[derive(Clone,Default, Debug)]
struct Pair([Elem; 2]);
type List<const LEN: usize> = List_<[Elem; LEN]>;
#[derive(Clone, Debug)]
struct List_<T: ?Sized>{ //NOTE secretly, LEN: u8
    tail: Index,
    used: u8,
    data: T,
}
impl<T: Copy> List_<[T]>{
    fn overwrite<const LEN: usize>(&mut self, other: &List_<[T; LEN]>){
        assert!(self.data.len() == other.data.len());
        self.tail = other.tail;
        self.used = other.used;
        self.data.copy_from_slice(&other.data);
    }
}

trait Meta { 
    fn len(&self)-> u8; fn list(&self)-> Option<&List_<[Elem]>>;
    fn fits(&self, ix: Index)-> bool {(0..self.len()).contains(&ix.2)}
    fn used(&self)-> u8 {if let Some(l) = self.list() {l.used} else {self.len()}}
    fn is_used(&self, ix: Index)-> bool {
        (0..self.used()).contains(&ix.2)
    }
}
impl Meta for Pair {
    fn len(&self)-> u8 {2}
    fn list(&self)->Option<&List_<[Elem]>> { None }
}
impl<const N: usize> Meta for List<N> {
    fn len(&self)-> u8 {N as u8}
    fn list(&self)->Option<&List_<[Elem]>> { Some(self) }
}

trait Data { 
    fn data(&self)-> &[Elem]; 
    fn get_elem(&self, ix: Index)-> &Elem { &self.data()[ix.2 as usize]}
}
impl Data for Pair { fn data(&self)->&[Elem] { &self.0 } }
impl<const T: usize> Data for List<T> {
    fn data(&self)->&[Elem] { &self.data }
}

trait DataMut: Data { fn data_mut(&mut self)-> &mut[Elem]; }
impl DataMut for Pair { fn data_mut(&mut self)->&mut[Elem] { &mut self.0 } }
impl<const T: usize> DataMut for List<T> {
    fn data_mut(&mut self)->&mut[Elem] { &mut self.data }
}

trait Use { fn use_idx(&mut self, i: u8);}
impl Use for Pair {fn use_idx(&mut self, _:u8){}}
impl<const T: usize> Use for List<T>{
    fn use_idx(&mut self, i:u8){
        assert!(i<=self.used, "i={} self={:?}", i, self);
        self.used=self.used.max(i+1)
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
enum PageType {Pair, List2, List6, List14} //, List30, List62, List126, List254}

trait Taggable { fn tag(&self)-> PageType; }
impl Taggable for Pair { fn tag(&self)->PageType { PageType::Pair } }
impl Taggable for List<2> { fn tag(&self)->PageType { PageType::List2 } }
impl Taggable for List<6> { fn tag(&self)->PageType { PageType::List6 } }
impl Taggable for List<14> { fn tag(&self)->PageType { PageType::List14 } }

trait Container: Meta + Data + DataMut + Taggable + Use + Debug {}
impl<T: Meta+Data+DataMut+Taggable+Use+Debug> Container for T {}

impl PageType {
    const fn max_idx(self)-> u8 {
        let size =  match self {
            PageType::Pair => PAGE_SIZE/mem::size_of::<Pair>(),
            PageType::List2 => PAGE_SIZE/mem::size_of::<List<2>>(),
            PageType::List6 => PAGE_SIZE/mem::size_of::<List<6>>(),
            PageType::List14 => PAGE_SIZE/mem::size_of::<List<14>>(),
        };
        if size-1 > u8::MAX as usize { panic!("TODO u10 page indexes") };
        (size-1) as u8
    }
    const fn next(self)-> Self {
        match self {
            PageType::Pair => PageType::List2, PageType::List2 => PageType::List6,
            PageType::List6 => PageType::List14, PageType::List14 => PageType::List14,
        }
    }
}

macro_rules! page_arr {
    ($T:ty) => {
      [[$T; PAGE_SIZE/mem::size_of::<$T>()]; PAGES]
    }
}

union Pages {
    pairs: page_arr!(Pair),
    list2: page_arr!(List<2>),
    list6: page_arr!(List<6>),
    list14: page_arr!(List<14>),
    /*list30: page_arr!(List<30>),
    list62: page_arr!(List<62>),
    list126: page_arr!(List<126>),
    list254: page_arr!(List<254>),*/
}

struct Store {
    free: [bool; PAGES], //FIXME bitvec
    full: [bool; PAGES], //FIXME bitvec
    types: [Option<PageType>; PAGES], //TODO this is an arrayvec honestly
    //FIXME u4 + hashtable backing? u8 + hashtable backing? also like sparsity / types mb
    rc: [[u16; PAGE_SIZE/mem::size_of::<Pair>()]; PAGES],
    pages: Pages,
}

impl core::ops::Index<Index> for Store {
    type Output = Elem;
    fn index(&self, ix: Index)-> &Elem { &self.grab(ix).unwrap().data()[ix.2 as usize] }
}

impl IndexMut<Index> for Store {
    fn index_mut(&mut self, ix: Index)-> &mut Elem {
        &mut self.grab_mut(ix).unwrap().data_mut()[ix.2 as usize]
    }
}

impl Store {
    fn alloc_page(&mut self, t: PageType) -> u16 {
        let i = self.free.iter().position(|x| *x).expect("OOM");
        self.free[i] = false;
        self.types[i] = Some(t);
        u16::try_from(i).unwrap()
    }
    fn alloc(&mut self, t: PageType)-> Index {
        for (page,full) in self.full.iter_mut().enumerate().filter(|(_,full)| !**full){
            match self.types[page] {
                None => {
                    self.free[page] = false;
                    self.types[page] = Some(t);
                    return self.gain(Index::of(page,0,0))
                },
                Some(tne) if tne != t => continue,
                Some(_) => {
                    let i: usize = self.rc[page].iter().position(|x|*x==0).expect("page was not full");
                    let last: usize = t.max_idx().into();
                    if !(0..=last).contains(&(i+1)) {*full=true};
                    return self.gain(Index::of(page,i,0))
                },
            }
        }
        panic!("OOM")
    }
    fn gain(&mut self, idx: Index)-> Index {
        let Index(page, list, _item) = idx;
        self.rc[page as usize][list as usize] += 1;
        // eprintln!("{} gain {:?}",page, self.rc[page as usize]);
        idx
    }
    // fn lose(&mut self, mut idx: Index) {
    //     loop {
    //         let Index(page, list, _item) = idx;
    //         self.rc[page as usize][list as usize] -= 1;
    //         if self.rc[page as usize][list as usize] > 0 { break; }
    //         //TODO fix used
    //         if let Some(Ok(idx)) = self.cdr(idx){} else { break; }
    //         //TODO free pages ever?
    //     }
    // }
    const fn new()-> Self {
        // let mut self_ =
        const W: usize = PAGE_SIZE/mem::size_of::<Pair>();
        const ZEROED: Pair = Pair([0,0]);
        Self {
            types: [None; PAGES],
            pages: Pages { pairs: [[ZEROED; W]; PAGES] }, // basically mem::zeroed()
            rc: [[0; W]; PAGES],
            free: [true; PAGES],
            full: [false; PAGES],
        }
    }
    // fn lookup(page:&[impl Container], at: usize) -> &impl Container {
    //     &page[at]
    // }
    fn grab(&self, ix: Index)-> Option<&dyn Container> { Some({
        let Index(page, list, _item) = ix;
        if self.rc[page as usize][list as usize] == 0 { None? };
        //SAFETY: as long as the types don't lie
        match self.types[page as usize]? {
            PageType::Pair => unsafe {&self.pages.pairs[page as usize][list as usize]},
            PageType::List2 => unsafe {&self.pages.list2[page as usize][list as usize]},
            PageType::List6 => unsafe {&self.pages.list6[page as usize][list as usize]},
            PageType::List14 => unsafe {&self.pages.list14[page as usize][list as usize]},
        }
      })
    }
    fn grab_mut(&mut self, ix: Index)-> Option<&mut dyn Container> { Some({
        let Index(page, list, _item) = ix;
        //SAFETY: as long as the types don't lie
        match self.types[page as usize]? {
            PageType::Pair => unsafe {&mut self.pages.pairs[page as usize][list as usize]},
            PageType::List2 => unsafe {&mut self.pages.list2[page as usize][list as usize]},
            PageType::List6 => unsafe {&mut self.pages.list6[page as usize][list as usize]},
            PageType::List14 => unsafe {&mut self.pages.list14[page as usize][list as usize]},
        }
      })
    }
    fn grab_mut_list(&mut self, ix: Index)-> Option<&mut List_<[Elem]>> { Some({
        let Index(page, list, _item) = ix;
        //SAFETY: as long as the types don't lie
        match self.types[page as usize]? {
            PageType::Pair => None?, // Not a list
            PageType::List2 => unsafe {&mut self.pages.list2[page as usize][list as usize]},
            PageType::List6 => unsafe {&mut self.pages.list6[page as usize][list as usize]},
            PageType::List14 => unsafe {&mut self.pages.list14[page as usize][list as usize]},
        }
      })
    }
    fn copy(&mut self, from: Index, to: Index){
        fn do_copy<const N: usize>(arr: &mut [[impl Clone; N]], from: Index, to: Index){
            arr[to.0 as usize][to.1 as usize] = arr[from.0 as usize][from.1 as usize].clone()
        }
        assert!(self.types[from.0 as usize] == self.types[to.0 as usize]);
        //SAFETY: as long as the types don't lie
        match self.types[from.0 as usize].unwrap() {
            PageType::Pair => unsafe {do_copy(&mut self.pages.pairs, from, to)},
            PageType::List2 => unsafe {do_copy(&mut self.pages.list2, from, to)},
            PageType::List6 => unsafe {do_copy(&mut self.pages.list6, from, to)},
            PageType::List14 => unsafe {do_copy(&mut self.pages.list14, from, to)},
        }
    }
    fn car(&self, ix: Index)-> Option<Elem>{
        let Index(_, _, item) = ix;
        let cell = self.grab(ix)?;
        if !cell.is_used(ix) { return None };
        if item == 0 { cell.list()?; }; // Pair has no tail
        Some(cell.data()[item as usize])
    }
    fn cdr(&self, ix: Index)-> Option<Result<Index,Elem>>{ Some({
        let Index(_, _, item) = ix;
        if item > 0 {Ok(ix - 1)}
        else {
            let cell = self.grab(ix)?;
            if !cell.is_used(ix) { None? };
            if let Some(list) = cell.list() { Ok(list.tail)}
            else { Err(cell.data()[0]) }
        }
    })}
    fn pair(&mut self, car: Elem, cdr: Elem)-> Index {
        let n = self.alloc(PageType::Pair);
        self[n] = cdr;
        self[n+1] = car;
        self.gain(n+1)
    }
    fn cons(&mut self, car: Elem, ix: Index)-> Option<Index>{ Some({
        // #[derive(Debug)] struct Cons(Elem,Index); eprint!("{:?} ",Cons(car,ix));
        let cdr = self.grab(ix)?;
        let next: Index = {
            if ix.2 < u8::MAX && cdr.fits(ix+1) && 
                    (!cdr.is_used(ix+1) || *cdr.get_elem(ix+1) == car) {
                self.gain(ix+1)
            } else if ix.2 == 0 && cdr.tag() == PageType::Pair {
                // NOTE in theory shouldn't happen, you'd just use the elem
                let n = self.alloc(PageType::Pair);
                self.copy(ix,n);
                n+1
            } else {
                let tag = cdr.tag().next();
                let n = self.alloc(tag);
                self.grab_mut_list(n).unwrap().tail = ix;
                n
            }
        };
        self[next] = car;
        self.grab_mut(next).unwrap().use_idx(next.2);
        next
    })}
    // fn empty(&self, mut ix: Index)-> Option<bool> {
    //     loop {
    //         if self.car(ix)? != 0 {
    //             return Some(false);
    //         }
    //         match self.cdr(ix){
    //             None => return Some(true),
    //             Some(iix) => ix = iix
    //         }
    //     }
    // }
    fn grab_page(&self, page: u16)-> Option<impl Iterator<Item=&dyn Container>+Clone> { Some({
        let p = page as usize;
        std::iter::empty()
        .chain((
            match self.types[p]? { 
                PageType::Pair => unsafe {&self.pages.pairs[p][..]}, _ => &[]
            }).iter().map(|x|->&dyn Container {x}))
        .chain((
            match self.types[p]? { 
                PageType::List2 => unsafe {&self.pages.list2[p][..]}, _ => &[]
            }).iter().map(|x|->&dyn Container {x}))
        .chain((
            match self.types[p]? { 
                PageType::List6 => unsafe {&self.pages.list6[p][..]}, _ => &[]
            }).iter().map(|x|->&dyn Container {x}))
        .chain((
            match self.types[p]? { 
                PageType::List14 => unsafe {&self.pages.list14[p][..]}, _ => &[]
            }).iter().map(|x|->&dyn Container {x}))
        // 
        //     PageType::List2 => unsafe {&self.pages.list2[page as usize].iter().map(|x|->&dyn Container {x})},
        //     PageType::List6 => unsafe {&self.pages.list6[page as usize].iter().map(|x|->&dyn Container {x})},
        // }
    })}
    fn get_iter(&self, i: Index) -> impl Iterator<Item=Elem> + Clone + '_ {
        IndexIterator {store: &self, idx:Some(i)}
    }
    fn non_free_pages(&self) -> impl Iterator<Item=usize> + Clone + '_ {
        self.free.iter().enumerate().filter(|(_,&x)| !x).map(|(i,_)|i)
    }
}

mod print {
    use core::fmt;
    use core::fmt::{Debug};

    pub struct Nullable<T>(pub Option<T>);
    impl<T:Debug> Debug for Nullable<T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            Ok(if let Some(x) = &self.0 { x.fmt(f)? })
        }
    }

    pub struct Listerator<D: Debug, I: Clone + Iterator<Item=D>>(pub I);
    impl<D: Debug, I: Clone + Iterator<Item=D>> Debug for Listerator<D,I> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_list().entries(self.0.clone()).finish()
        }
    }

    pub struct Maperator<K: Debug, V: Debug, I: Clone + Iterator<Item=(K,V)>>(pub I);
    impl<K: Debug, V: Debug, I: Clone + Iterator<Item=(K,V)>> Debug for Maperator<K,V,I> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_map().entries(self.0.clone()).finish()
        }
    }

    pub struct Lined<T>(pub T);
    impl<T: Debug> Debug for Lined<T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            self.0.fmt(f)?;
            write!(f,"\n")
        }
    }
    impl<T: Clone> Clone for Lined<T> {fn clone(&self) -> Self { Self(self.0.clone())}}
}

#[derive(Clone)]
struct IndexIterator<'a> {
    store: &'a Store,
    idx: Option<Index>
}
impl Iterator for IndexIterator<'_> {
    type Item = Elem;
    fn next(&mut self)-> Option<Elem>{
        let a = self.store.car(self.idx?);
        let idx = self.store.cdr(self.idx?)?;
        self.idx = idx.ok();
        match (a,idx) {
            (Some(_), Ok(_))=> a,
            (None, Err(e))=> Some(e),
            _ => panic!("yo")
        }
    }
}

#[derive(Debug)]
struct DebugRow<'a, T: Debug + Sized>{
    ty: PageType,
    rc: &'a [u16],
    x: T,
}

impl Debug for Store {
    fn fmt<'a>(&'a self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(
            self.non_free_pages().map(|i|{
                let page = self.grab_page(i as u16).unwrap();
                let page = page.enumerate().filter(move |(j,_)| self.rc[i][*j] != 0);
                (i, print::Lined(DebugRow {
                    ty: self.types[i].unwrap(),
                    rc: &self.rc[i],
                    x: print::Maperator(page)
                }))
            })
        ).finish()
    }
}
struct Elems<'a>(&'a Store);
// impl AsRef<Store> for Elems<'_> { fn as_ref(&self)-> &Store {&self.0}}
impl Debug for Elems<'_> {
    fn fmt<'a>(&'a self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use print::{Lined, Listerator, Maperator};
        Maperator(
            // (&[(1,2)]).iter().cloned()
            self.0.non_free_pages().flat_map(|i|{
                let page = self.0.grab_page(i as u16).unwrap();
                let page = (0..page.count()).into_iter().filter(move |j| self.0.rc[i][*j] != 0);
                page.flat_map(move |j|{
                    let max = self.0.grab(Index::of(i,j,0)).unwrap().used().into();
                    (0..max).into_iter().map(
                        move |k| (Index::of(i,j,k), Lined(Listerator(self.0.get_iter(Index::of(i,j,k)))))
                    )
                })
            })
        ).fmt(f)
    }
}


#[cfg(test)]
mod tests {
    use crate::*;
    use print::Listerator;

    macro_rules! nodbg { ($e:expr)=> {$e} }
    fn _size<T>(_:T)-> usize { mem::size_of::<T>()}

    #[test]
    fn main(){
        // dbg!(mem::size_of::<PageType>());
        let mut store = Box::new(Store::new());
        nodbg!(store.alloc_page(PageType::Pair));
        dbg!(store.alloc_page(PageType::List2));
        dbg!(store.pair(2,1));
        if true {
            dbg!(store.car(Index(0,0,1)),store.cdr(Index(0,0,1)));
            dbg!(store.car(Index(0,0,0)),store.cdr(Index(0,0,0)));
        }
        for _ in 1..10 {
            store.alloc_page(PageType::Pair);
        }
        {
        dbg!(store.alloc_page(PageType::List2));
        dbg!(store.cons(2, Index(0,0,0)));
        let pairs = format!{"{:?}", unsafe {&store.pages.pairs[0][..4]}}; dbg!(pairs);
        let c3 = dbg!(store.cons(3, Index(0,0,0))).unwrap();
        let pairs = format!{"{:?}", unsafe {&store.pages.pairs[0][..4]}}; dbg!(pairs);
        dbg!(Listerator(store.get_iter(c3)));
        let c4 = dbg!(store.cons(4, c3)).unwrap();
        dbg!(Listerator(store.get_iter(c4)), store.grab(c4));
        // if true {return ()};
        let pairs = format!{"{:?}", unsafe {&store.pages.pairs[0][..4]}}; dbg!(pairs);
        dbg!(store.alloc(PageType::Pair));
        nodbg!(unsafe{store.pages.pairs.len()});
        // dbg!(store.empty(Index(0,0,0)));
        // dbg!({let l = store.pair(0,0); store.empty(l)});
        // dbg!(store.empty(Index(1,0,0)));
        dbg!(format!{"{:?}", &store.rc[0]});
        
        let l6 = nodbg!(store.alloc(PageType::List6)); 
        let l6d = List::<6>{ tail: c3, used: 3, data: [4,5,6,77,0,0]};
        store.grab_mut_list(l6).unwrap().overwrite(&l6d);
        
        // dbg!(store.grab(Index(0,0,0)), store.grab(Index(0,4,0)), store.grab(Index(1,0,2)));
        // dbg!(size(store.types));
        // dbg!(mem::size_of::<Index>());
        // dbg!([PageType::Pair, PageType::List2, PageType::List6].map(|a|{mem::discriminant(&a)}));
        store.gain(Index(3,3,0)); unsafe {store.pages.pairs[3][3] = Pair([13,13])};
        }
        // eprintln!("{:?}",store);
        // eprintln!("{:?}",Elems(&store));
        {
            let mut l = store.pair(18,19);
            for i in (1..=17).rev() {
                l = store.cons(i,l).unwrap()
            }
            eprintln!("l = {:?}",Listerator(store.get_iter(l)));
        }
        println!("{:?}",store);
        println!("{:?}",Elems(&store));
    }
}

// TODO
// - free
// - property-test allocating and reading various nouns
//   + memory consumption of intercepting a list in the middle

mod nock {
    use crate::print::Listerator;
    use std::{mem, iter::FromIterator, collections::VecDeque};
    use crate::{Elem, Store, Index};
    type Axis = u32;
    type Offset = usize;

    #[derive(Copy, Clone, Debug)]
    #[repr(u8)]
    enum Op {
        Dup = 13, Exch = 14, Cons = 12,      // d e c
        If(Offset) = 6, Fwd(Offset) = 15,    // 6 f
        Nok = 2, Cel = 3, Inc = 4, Eql = 5,  // 2 3 4 5
        Get(Axis) = 0, Lit(Elem) = 1,        // 0 1
        Pin = 8, Run(Axis) = 9,              // 8 9
        Hint = 10, Thin = 7,                 // a 7
        Scry = 11,                           // b
    }

    #[derive(Debug)]
    enum Noun {
        Dir(u32),
        Ind(Index),
        Cel(Index),
    }

    impl From<Elem> for Noun {
        fn from(e: Elem)-> Noun {
            match e >> 30 {
                2 => Noun::Cel((e & 0x3fff_ffff).into()),
                3 => Noun::Ind((e & 0x3fff_ffff).into()),
                _ => Noun::Dir(e),
            }
        }
    }
    impl Into<Elem> for Noun {
        fn into(self)-> Elem {
            match self {
                Noun::Dir(a) => a,
                Noun::Ind(i) => 0xa000_0000u32 | Into::<u32>::into(i),
                Noun::Cel(i) => 0x8000_0000u32 | Into::<u32>::into(i),
            }
        }
    }
    
    //TODO wrapper around Elem that has a Drop and Clone lol
    fn nybble(mut subj: Elem, code: &[Op]) -> Elem {
        let code = VecDeque::from_iter(code.into_iter().copied());
        let mut retn: Vec<VecDeque<Op>> = vec![code];
        //
        const CONS: Elem = 0xcccc_cccc;
        const SNOC: Elem = 0xcccc_dddd;
        let mut stack: Vec<Elem> = vec![];
        let heap: &mut Store = global_store();

        fn lose(stack: &mut Vec<Elem>, subj: Elem){
            //TODO actual lose
            if ![CONS,SNOC].contains(&subj) { return }
            let mut deep = 1u32;
            while deep > 0 {
                if ![CONS,SNOC].contains(&stack.pop().expect("Cons-free underflow")) {
                    deep += 1
                }
                else { deep -= 1; }
            }
        }
        fn reify_cons(heap: &mut Store, stack: &mut Vec<Elem>, subj: &mut Elem){
            if ![CONS,SNOC].contains(subj) { return }
            // println!("CONS/SNOC {:?}", Listerator(stack.iter().rev().take(2)));
            let op = *subj; *subj = stack.pop().unwrap();
            reify_cons(heap, stack, subj); //head
            let pop = *subj; *subj = stack.pop().expect("Cons underflow");
            reify_cons(heap, stack, subj);
            let (car, cdr) = match op {
                CONS => (pop, *subj),
                SNOC => (*subj, pop),
                _ => panic!()
            };
            let cel = match cdr.into() {
                Noun::Cel(ix) => heap.cons(car,ix).expect("Bad index"),
                _ => heap.pair(car, cdr),
            };
            // println!("pop={} subj={}, car={} cdr={} -> cel={:?} {}", pop, subj, car, cdr, cel, Into::<Elem>::into(Noun::Cel(cel)));
            *subj = Noun::Cel(cel).into()
        }
        while let Some(mut code) = retn.pop(){
            while let Some(inst) = code.pop_front() {
                if false {println!("{:?} {:?} {:?}",
                    inst,
                    Listerator(stack.iter().map(|x| Noun::from(*x))),
                    Noun::from(subj)
                );}
                let mut do_cons = || { reify_cons(heap, &mut stack, &mut subj) };
                match inst {
                    Op::Dup => { do_cons(); stack.push(subj) },
                    Op::Exch => {
                        do_cons(); //TODO C 1 C 2 3
                        mem::swap(&mut subj, stack.last_mut().unwrap())
                    },
                    Op::Pin => { stack.push(subj); subj = CONS },
                    Op::Cons => { stack.push(subj); subj = SNOC },
                    //
                    Op::Fwd(ofs) => {code.drain(..ofs);},
                    Op::If(ofs) => if as_bool(subj) { code.drain(..ofs); },
                    //
                    Op::Nok => {
                        retn.push(code); code = compile(subj);
                        subj = stack.pop().unwrap();
                    }
                    Op::Cel => {
                        lose(&mut stack, subj); //TODO drop shenanigans
                        subj = loobean(is_cell(subj) || subj == CONS || subj == SNOC)
                    },
                    Op::Inc => { assert!(is_atom(subj)); subj += 1 }, //TODO indirect
                    Op::Eql => {
                        do_cons();
                        let to = stack.pop().unwrap();
                        reify_cons(heap, &mut stack, &mut subj);
                        subj = loobean(subj == to || unify(subj,to)) //TODO lose
                    }
                    Op::Lit(e) => { lose(&mut stack, subj); subj = e}
                    Op::Get(ax) => {
                        do_cons(); //TODO or navigate them properly
                        //TODO lose rest
                        subj = cdadr(heap, subj, ax).expect("Bad axis")
                    }
                    Op::Run(ax) => {
                        do_cons(); //TODO or navigate them properly
                        let call = cdadr(heap, subj, ax).expect("Bad axis"); //TODO cons
                        retn.push(code); code = compile(call);
                    }
                    Op::Hint => {
                        do_cons();
                        println!("hint: {:?}", Noun::from(subj));
                        subj = stack.pop().unwrap() //TODO
                    }
                    Op::Thin => {}
                    Op::Scry => {panic!("No scry handler")}
                }
            }
        }
        reify_cons(heap, &mut stack, &mut subj);
        assert!(stack.is_empty(), "Left items on stack: {:?} {:?}", 
            Listerator(stack.iter().map(|x| Noun::from(*x))),
            Noun::from(subj)
        );
        match subj.into() {
            Noun::Dir(x) => println!("{}", x),
            Noun::Cel(ix) | Noun::Ind(ix) => {
                println!("{:?}", Listerator(heap.get_iter(ix)))
            }
        }
        return subj;
    }
    
    fn global_store()-> &'static mut Store { Box::leak(Box::new(Store::new())) } //TODO reuse
    fn is_cell(e: Elem)-> bool { match e.into() { Noun::Cel(_) => true, _ => false } }
    fn is_atom(e: Elem)-> bool { !is_cell(e) }

    fn as_bool(e: Elem)-> bool { 
        assert!(is_atom(e));
        assert!(e <= 1);
        return (e == 0)
    }
    fn loobean(b: bool)-> Elem { if b { 0 } else { 1 } }

    fn compile(e: Elem) -> VecDeque<Op> { unimplemented!(); VecDeque::new() }
    fn unify(a: Elem, b: Elem) -> bool { false /*TODO*/ }
    fn cdadr(mem: &Store, mut e: Elem, mut ax: Axis) -> Option<Elem> {
        assert!(ax != 0);
        //TODO indirect
        let mut a = Noun::from(e);
        for bit in (0..32usize).into_iter().rev()
                         .skip_while(|bit| 0 == ax & 1<<bit).skip(1) {
            if let Noun::Cel(idx) = a {
                if 0 == ax & 1<<bit {
                    a = mem.car(idx)?.into()
                } else {
                    let idx = mem.cdr(idx)?.ok()?;
                    if let Some(Err(n)) = mem.cdr(idx) { // fixup last-atom ambiguity
                        a = Noun::Dir(n)
                    } else {
                        a = Noun::Cel(idx)
                    }
                } 
            } else { None? }
        }
        return Some(a.into());
    }
    fn encode(a: Result<Index,Elem>) -> Elem { unimplemented!()}

    #[test]
    fn dup_inc(){
        use Op::*;
        println!("{:?}", Noun::from(nybble(1, &[Dup, Inc, Cons, Dup, Lit(0), Pin])));
    }
}

//   [ 1 . [ 2 . 3 ] ]
// 0 0 1 0 0 2 0 3 S S
//   0 0 1 0 0 2 2 3 S
//         1 1 1 1 2 3
//                 1 2
//                   1

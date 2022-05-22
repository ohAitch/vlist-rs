#![feature(arbitrary_enum_discriminant)]
#![allow(incomplete_features)] #![feature(inline_const)]
#![feature(untagged_unions)]
#![feature(core_intrinsics)]

#![allow(unused, dead_code)] // TODO

#![allow(clippy::from_over_into)]
#![allow(clippy::unit_arg)]
#![allow(clippy::precedence)] // :)

use arrayvec::ArrayVec;

use std::convert::{TryInto,TryFrom};
use std::iter::{Peekable};
use std::mem;

use core::fmt;
use core::fmt::{Debug};
use core::ops::{IndexMut};

mod nock;

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

//TODO 4096 maybe  L: this should be a test case - allocate a large thing that doesn't fit
const PAGE_SIZE: usize = 128;
const PAGES: usize = 1024;
type Elem = u32;

#[derive(Clone, Copy, PartialEq, Eq)]
struct Index(u16,u8,u8);
impl Index {
    fn of(a:usize, b:usize, c:usize)-> Self { Self(a.try_into().unwrap(), b.try_into().unwrap(), c.try_into().unwrap())}
    fn of32(a:u32, b:u32, c:u32)-> Self { Self(a.try_into().unwrap(), b.try_into().unwrap(), c.try_into().unwrap())}
}
impl std::ops::Add<u8> for Index {
    type Output = Self; fn add(self, ofs: u8)-> Self {
      Self(self.0, self.1, self.2 + ofs)
    }
}

impl std::ops::Sub<u8> for Index {
    type Output = Self; fn sub(self, ofs: u8)-> Self {
      Self(self.0, self.1, self.2 - ofs)
    }
}

impl Debug for Index {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f,"{}-{}-{}", self.0, self.1, self.2)
    }
}

impl Into<Elem> for Index {
  fn into(self) -> Elem {
    (self.0 as u32)*0x1_0000 ^ (self.1 as u32)*0x100 ^ (self.2 as u32)
  }
}

impl From<Elem> for Index {
  fn from(e: Elem) -> Self {
    Self::of32(e>>16, e>>8 & 0xff, e&0xff)
  }
}

trait StoreCloneable {
  fn duplicate(&self) -> Self;
}

#[derive(Default, Debug)]
struct Pair([Elem; 2]);

impl StoreCloneable for Pair {
  fn duplicate(&self) -> Self {
    Pair([self.0[0], self.0[1]])
  }
}


type List<const LEN: usize> = List_<[Elem; LEN]>;


#[derive(Debug)]
struct List_<T: ?Sized>{ //NOTE secretly, LEN: u8
    tail: Index,
    used: u8,
    data: T,
}

impl<T: Copy> StoreCloneable for List_<T> {
  fn duplicate(&self) -> Self {
    Self {
      tail: self.tail,
      used: self.used,
      data: self.data,
    }
  }
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
    fn use_idx(&mut self, i:u8) {
        assert!(i<=self.used, "i={} self={:?}", i, self);
        self.used = self.used.max(i+1)
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
    /// Max number of items that can fit into a page. This is the same as
    /// PageType::Pair.items_per_page() because a Pair is the smallest-sized item that can be
    /// stored in a page.
    const fn max_items() -> usize {
      PageType::Pair.items_per_page()
    }

    const fn items_per_page(&self) -> usize {
      match self {
        PageType::Pair => PAGE_SIZE/mem::size_of::<Pair>(),
        PageType::List2 => PAGE_SIZE/mem::size_of::<List<2>>(),
        PageType::List6 => PAGE_SIZE/mem::size_of::<List<6>>(),
        PageType::List14 => PAGE_SIZE/mem::size_of::<List<14>>(),
      }
    }

    const fn max_idx(self)-> u8 {
        let size = self.items_per_page();
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
    rc: [[u16; PageType::max_items()]; PAGES],
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
    fn lose(&mut self, mut idx: Index) {
        loop {
            let Index(page, list, _item) = idx;
            self.rc[page as usize][list as usize] -= 1;
            if self.rc[page as usize][list as usize] > 0 { break; }
            //TODO fix used
            if let Some(Ok(idx)) = self.cdr(idx){} else { break; }
            //TODO free pages ever?
        }
    }
    const fn new()-> Self {
        const W: usize = PageType::Pair.items_per_page();
        const ZEROED: [Pair; W]  = [const{Pair([0,0])}; W];
        Self {
            types: [None; PAGES],
            pages: Pages { pairs: [ZEROED; PAGES] }, // basically mem::zeroed()
            rc: [[0; PageType::max_items()]; PAGES],
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
        fn do_copy<const N: usize>(arr: &mut [[impl StoreCloneable; N]], from: Index, to: Index){
            arr[to.0 as usize][to.1 as usize] = arr[from.0 as usize][from.1 as usize].duplicate()
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
                self.copy(ix, n);
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
    fn buffer(&mut self, vals: &[Elem])-> Index{
        if vals.len() < 2 {
            self.pair(*vals.first().unwrap_or(&0), 0)
        } else {
            let mut vals = vals.iter().rev().copied();
            let cdr = vals.next().unwrap();
            let mut ix = self.pair(vals.next().unwrap(), cdr);
            for i in vals { ix = self.cons(i, ix).unwrap()}
            ix
        }
    }
    fn buffer_bytes(&mut self, vals: &[u8])-> Index {
        if let Some(0) = vals.last() {
            panic!("Cannot represent trailing 0 bytes")
        }
        //TODO implementing `buffer` in terms of iterators would be possible but annoying
        let words: Vec<u32> = vals.chunks(4).map(|b| {
            let mut c = [0u8; 4];
            c[..b.len()].copy_from_slice(b);
            u32::from_le_bytes(c)
        }).collect();
        self.buffer(&words)
    }

    fn grab_page(&self, page: u16)-> Option<impl Iterator<Item=&dyn Container>+Clone> {
        let page = page as usize;
        let ty = self.types[page]?;
        return Some(PageIterator {
          idx: 0,
          ty,
          page,
          store: self,
        });

        #[derive(Clone)]
        struct PageIterator<'a> {
          idx: usize,
          ty: PageType,
          page: usize,
          store: &'a Store
        }

        impl<'a> Iterator for PageIterator<'a> {
          type Item = &'a dyn Container;
          fn next(&mut self) -> Option<Self::Item> {
            unsafe {
              let ptr: &dyn Container = match self.ty {
                PageType::Pair => self.store.pages.pairs[self.page].get(self.idx)? as &dyn Container,
                PageType::List2 => self.store.pages.list2[self.page].get(self.idx)? as &dyn Container,
                PageType::List6 => self.store.pages.list6[self.page].get(self.idx)? as &dyn Container,
                PageType::List14 => self.store.pages.list14[self.page].get(self.idx)? as &dyn Container,
              };
              self.idx += 1;
              Some(ptr)
            }
          }
        }
    }

    fn get_iter(&self, i: Index) -> impl Iterator<Item=Elem> + Clone + '_ {
        IndexIterator {store: self, idx:Some(i)}
    }
    fn get_iter_bytes(&self, i: Index) -> impl Iterator<Item=u8> + Clone + '_ {
        ElemBytes(self.get_iter(i).peekable()).flatten()
    }
    fn non_free_pages(&self) -> impl Iterator<Item=usize> + Clone + '_ {
        self.free.iter().enumerate().filter(|(_,&x)| !x).map(|(i,_)|i)
    }
    fn used_bytes(&self) -> usize {
        unimplemented!();
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
            writeln!(f)
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
        let cur_idx = self.idx?;

        let a = self.store.car(cur_idx);
        let idx = self.store.cdr(cur_idx)?;
        self.idx = idx.ok();
        match (a, idx) {
            (Some(_), Ok(_))=> a,
            (None, Err(e))=> Some(e),
            _ => panic!("yo")
        }
    }
}

#[derive(Clone)]
struct ElemBytes<I: Iterator<Item=Elem>>(Peekable<I>);
impl<I> Iterator for ElemBytes<I> where I: Iterator<Item=Elem> {
    type Item = ArrayVec<u8,4>;
    fn next(&mut self)-> Option<ArrayVec<u8,4>>{
        let mut bytes: ArrayVec<u8, 4> = self.0.next()?.to_le_bytes().into();
        if self.0.peek().is_none() { // no further elems
            let last_nonzero = bytes.iter().rposition(|x| *x != 0 );
            let trailing = 1 + last_nonzero.expect("Trailing zero word");
            bytes.truncate(trailing) // drop zero bytes at end of buffer
        };
        Some(bytes)
    }
}


#[derive(Debug)]
struct Row<'a, T: Debug + Sized>{
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
                (i, print::Lined(Row {
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
    use matches::assert_matches;

    macro_rules! nodbg { ($e:expr)=> {$e} }
    fn _size<T>(_:T)-> usize { mem::size_of::<T>()}

    #[test]
    fn page_allocation_and_refcounts() {
      let mut store = Box::new(Store::new());
      assert_eq!(0, store.alloc_page(PageType::Pair));
      assert_eq!(1, store.alloc_page(PageType::List2));
      assert_eq!(2, store.alloc_page(PageType::List6));

      assert_eq!(store.rc[0][0], 0);
      let idx = store.alloc(PageType::Pair);
      assert_eq!(idx, Index(0, 0, 0));
      assert_eq!(store.rc[0][0], 1);

      let idx = store.alloc(PageType::Pair);
      assert_eq!(idx, Index(0, 1, 0));
      assert_eq!(store.rc[0][0], 1);
      assert_eq!(store.rc[0][1], 1);

      let idx = store.alloc(PageType::List2);
      assert_eq!(idx, Index(1, 0, 0));
      assert_eq!(store.rc[1][0], 1);

      let idx = store.alloc(PageType::List6);
      assert_eq!(idx, Index(2, 0, 0));
      assert_eq!(store.rc[2][0], 1);
    }

    #[test]
    fn allocate_on_next_free_page_when_page_full() {
      let mut store = Box::new(Store::new());
      let total = PageType::Pair.items_per_page();

      let idx = store.alloc(PageType::Pair);
      assert_eq!(idx, Index(0, 0, 0));

      store.alloc_page(PageType::List6);

      for n in 1..total {
        let idx = store.alloc(PageType::Pair);
        assert_eq!(idx, Index::of(0, n, 0));
      }

      let idx = store.alloc(PageType::Pair);
      assert_eq!(idx, Index(2, 0, 0));
    }

    #[test]
    fn allocate_pairs() {
      let mut store = Box::new(Store::new());
      let idx = store.pair(1, 2);
      assert_eq!(idx, Index(0, 0, 1));
      let idx = store.pair(4, 10);
      assert_eq!(idx, Index(0, 1, 1));

      unsafe {
        let p = store.pages.pairs.as_ptr() as *const u32;
        assert_eq!(*p, 0x2);
        assert_eq!(*p.add(1), 0x1);
        assert_eq!(*p.add(2), 0xa);
        assert_eq!(*p.add(3), 0x4);
      }
    }

    #[test]
    fn allocate_cons() {
      let mut store = Box::new(Store::new());
      let idx = store.pair(14, 15);

      let cons_idx = store.cons(13, idx).unwrap();
      assert_eq!(cons_idx, Index(1, 0, 0));
      assert_matches!(store.types[1], Some(PageType::List2));

      let cons_idx_2 = store.cons(12, cons_idx).unwrap();
      assert_eq!(cons_idx_2, Index(1, 0, 1));

      let cons_idx_3 = store.cons(11, cons_idx_2).unwrap();
      assert_eq!(cons_idx_3, Index(2, 0, 0));
      assert_matches!(store.types[2], Some(PageType::List6));

      let cons_idx_4 = store.cons(10, cons_idx_3).unwrap();
      assert_eq!(cons_idx_4, Index(2, 0, 1));

      let start_list2 = unsafe { &store.pages.list2 };
      let p = &start_list2[1][0];
      assert_eq!(p.tail, Index(0, 0, 1));
      assert_matches!(p.data, [13, 12]);

      let start_list6 = unsafe { &store.pages.list6 };
      let p = &start_list6[2][0];
      assert_eq!(p.tail, Index(1, 0, 1));
      assert_matches!(p.data, [11, 10, 0, 0, 0, 0]);
    }

    // Mostly copied from the first part of main()
    #[test]
    fn alloc_session_1() {
      let mut store = Box::new(Store::new());

      assert_eq!(0, store.alloc_page(PageType::Pair));
      assert_eq!(1, store.alloc_page(PageType::List2));
      let pair_idx = Index(0, 0, 1);
      assert_eq!(pair_idx, store.pair(2, 1));

      assert_eq!(Some(2), store.car(pair_idx));
      assert_eq!(Some(Ok(Index(0, 0, 0))), store.cdr(pair_idx));

      let zero_idx = Index(0, 0, 0);
      assert_eq!(None, store.car(zero_idx));
      assert_eq!(Some(Err(1)), store.cdr(zero_idx));

      for _ in 1..10 {
        store.alloc_page(PageType::Pair);
      }

      assert_eq!(11, store.alloc_page(PageType::List2));
      assert_eq!(Some(Index(0, 0, 1)), store.cons(2, Index(0, 0, 0)));

      let pairs = unsafe { &store.pages.pairs[0][..4] };
      assert_matches!(pairs, [
          Pair([1, 2]),
          Pair([0, 0]),
          Pair([0, 0]),
          Pair([0, 0]),
      ]);

      let c3 = store.cons(3, Index(0, 0, 0)).unwrap();
      assert_eq!(c3, Index(0, 1, 1));

      let pairs = unsafe { &store.pages.pairs[0][..4] };
      assert_matches!(pairs, [
          Pair([1, 2]),
          Pair([1, 3]),
          Pair([0, 0]),
          Pair([0, 0]),
      ]);

      let c3_elems: Vec<Elem> = store.get_iter(c3).collect();
      assert_eq!(c3_elems.as_slice(), [3, 1]);

      let c4 = store.cons(4, c3).unwrap();
      assert_eq!(c4, Index(1, 0, 0));

      let c4_elems: Vec<Elem> = store.get_iter(c4).collect();
      assert_matches!(c4_elems.as_slice(), [4, 3, 1]);

      let pairs = unsafe { &store.pages.pairs[0][..4] };
      assert_matches!(pairs, [
          Pair([1, 2]),
          Pair([1, 3]),
          Pair([0, 0]),
          Pair([0, 0]),
      ]);

      let idx = store.alloc(PageType::Pair);
      assert_eq!(idx, Index(0, 2, 0));

      let reference_count = store.rc[0];
      assert_matches!(reference_count, [3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn buffer_test() {
      let mut store = Box::new(Store::new());

      let buffer_idx = store.buffer(&(1..=19).collect::<Vec<u32>>());
      let buffer_elems: Vec<Elem> = store.get_iter(buffer_idx).collect();
      assert_matches!(buffer_elems[..],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
      );
    }

    #[test]
    fn string_test() {
      let mut store = Box::new(Store::new());

      let buffer: Index = store.buffer_bytes(b"abcde");
      let bytes: Vec<u8> = store.get_iter_bytes(buffer).collect::<Vec<_>>();
      assert_eq!(b"abcde", bytes.as_slice());
    }

    #[test]
    fn main(){
        // dbg!(mem::size_of::<PageType>());
        let mut store = Box::new(Store::new());
        dbg!(store.alloc_page(PageType::Pair));
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
            // let mut l = store.pair(18,19);
            // for i in (1..=17).rev() {
            //     l = store.cons(i,l).unwrap()
            // }
            let l = store.buffer(&(1..=19).collect::<Vec<u32>>());
            eprintln!("l = {:?}",Listerator(store.get_iter(l)));
        }
        println!("{:?}",store);
        println!("{:?}",Elems(&store));
    }
}


//   [ 1 . [ 2 . 3 ] ]
// 0 0 1 0 0 2 0 3 S S
//   0 0 1 0 0 2 2 3 S
//         1 1 1 1 2 3
//                 1 2
//                   1

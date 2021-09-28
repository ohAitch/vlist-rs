// TODO
// - free
// - property-test allocating and reading various nouns
//   + memory consumption of intercepting a list in the middle

use itertools::Itertools; // 0.9.0
use ::slice_of_array::prelude::*;

use crate::{Elems, print::Listerator};
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

impl Op {
    fn arg(self)-> Option<Elem> { Some({
        use Op::*;
        match self {
            Get(x) | Run(x) | Lit(x) => x,
            If(x) | Fwd(x) => x as Elem,
            _ => None?
        }
    })}

    fn enc(&self)-> [Elem; 2] {
        [ std::intrinsics::discriminant_value(self) as Elem,
          self.arg().unwrap_or(99)
        ]
    }
}

#[derive(Debug, PartialEq, Eq)]
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
            Noun::Ind(i) => 0x3u32 << 30 | Into::<u32>::into(i),
            Noun::Cel(i) => 0x2u32 << 30 | Into::<u32>::into(i),
        }
    }
}

//TODO wrapper around Elem that has a Drop and Clone lol
fn nybble(heap: &mut Store, mut subj: Elem, code: &[Op]) -> Elem {
    let code = VecDeque::from_iter(code.iter().copied());
    let mut retn: Vec<VecDeque<Op>> = vec![code];
    //
    const CONS: Elem = 0xcccc_cccc;
    const SNOC: Elem = 0xcccc_dddd;
    let mut stack: Vec<Elem> = vec![];

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
                    retn.push(code); code = compile(heap,subj);
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
                    subj = axe(heap, subj, ax).expect("Axed atom")
                }
                Op::Run(ax) => {
                    do_cons(); //TODO or navigate them properly
                    let call = axe(heap, subj, ax).expect("Axed atom"); //TODO cons
                    retn.push(code); code = compile(heap,call);
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
    subj
}

fn is_cell(e: Elem)-> bool { matches!(e.into(), Noun::Cel(_)) }
fn is_atom(e: Elem)-> bool { !is_cell(e) }

fn as_bool(e: Elem)-> bool {
    assert!(is_atom(e));
    assert!(e <= 1);
    (e == 0)
}
fn loobean(b: bool)-> Elem { if b { 0 } else { 1 } }

fn compile(heap: &Store, e: Elem) -> VecDeque<Op> {
    let ix = if let Noun::Ind(ix) = e.into() {ix} else {panic!("Expected bytecode {:?}", Noun::from(e))};
    heap.get_iter(ix).tuples().map(|(op, arg): (Elem, Elem)| -> Op {
        use Op::*;
        match op {
            2 => Nok, 3 => Cel, 4 => Inc, 5 => Eql,
            7 => Thin, 8 => Pin, 10 => Hint,
            11 => Scry, 12 => Cons, 13 => Dup, 14 => Exch,
            6 => If(arg as usize), 15 => Fwd(arg as usize),
            0 => Get(arg), 1 => Lit(arg), 9 => Run(arg),
            _ => panic!("Op out of range")
        }
    }).collect()
}
fn unify(a: Elem, b: Elem) -> bool { false /*TODO*/ }
fn axe(mem: &Store, mut e: Elem, mut ax: Axis) -> Option<Elem> {
    assert!(ax != 0);
    //TODO indirect
    let mut a = Noun::from(e);
    for bit in (0..32usize).into_iter().rev()
                      .skip_while(|bit| 0 == ax & 1<<bit).skip(1) {
        dbg!(&a, ax, bit, ax & 1<<bit);
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
    Some(a.into())
}
fn encode(a: Result<Index,Elem>) -> Elem { unimplemented!()}

fn heap() -> &'static mut Store {Box::leak(Box::new(Store::new()))}


mod test {
    use super::*;
    #[test]
    fn dup_inc(){
        use Op::*;
        assert_eq!(Noun::Cel(Index(0,0,1)), Noun::from(
            nybble(heap(), 1, &[Dup, Inc, Cons])
        ));
    }

    #[test]
    fn three(){
        use Op::*;
        let store = heap();
        assert_eq!(Noun::Cel(Index(1,0,0)),
            Noun::from(nybble(store, 1, &[Dup, Inc, Cons, Dup, Lit(0), Pin]))
        );
        // println!("{:?}",Elems(&store));
    }

    #[test]
    #[should_panic(expected = "Axed atom")]
    fn axe_atom(){
        use Op::*;
        nybble(heap(), 1, &[Get(123)]);
    }

    #[test]
    fn axe(){
        use Op::*;
        assert_eq!(Noun::Dir(1), Noun::from(nybble(heap(), 1, &[Dup, Inc, Cons, Dup, Cons, Get(6)])));
        assert_eq!(Noun::Dir(2), Noun::from(nybble(heap(), 1, &[Dup, Inc, Cons, Dup, Cons, Get(2), Get(3)])));
    }

    #[test]
    fn noun_enc(){
        assert_eq!(Noun::Dir(10), 10.into());
        assert_eq!(10u32, Noun::Dir(10).into());
        let ix = Index(1,2,3);
        assert_eq!(Noun::Cel(ix), From::<Elem>::from(Noun::Cel(ix).into()));
        assert_eq!(Noun::Ind(ix), From::<Elem>::from(Noun::Ind(ix).into()));
    }

    #[test]
    fn decode(){
        use std::intrinsics::discriminant_value;
        fn tag(o: Op)-> Elem {discriminant_value(&o) as Elem}
        let store = heap();

        let code = &[
            0,100, 1,101, 2,99, 3,99, 4,99, 5,99, 6,106, 7,99,
            8,99, 9,109, 10,99, 11,99, 12,99, 13,99, 14,99, 15,115
        ];
        let atom = store.buffer(code);
        println!("{:?}", Listerator(store.get_iter(atom)));
        let ops = compile(store, Noun::Ind(atom).into());
        println!("{:?}", ops);
        assert_eq!(ops.len() * 2, code.len());
        for (i,op) in ops.iter().enumerate() {
            assert_eq!(i, discriminant_value(op) as usize);
            assert_eq!(code[2*i+1], op.arg().unwrap_or(99));
        }
    }

    fn program<'a, B: Copy + IntoIterator<Item=&'a Op>>(a: &mut Store, b: B) -> Elem {
        Noun::Ind(
            a.buffer(b.into_iter().map(|x|x.enc()).collect::<Vec<_>>().flat())
        ).into()
    }
    #[test]
    fn exec(){
        use Op::*;
        let store = heap();
        // println!("{:?}",Elems(&store));
        let prog = program(store,&[Dup, Lit(1), Inc, Inc, Inc, Pin]);
        // println!("{:?}", Listerator(store.get_iter(match prog.into() { Noun::Ind(x) => x, _=> panic!()})));
        let output = nybble(store, prog, &[Run(1)]).into();
        if let Noun::Cel(ix) = output {
            assert_eq!(vec![4, prog], store.get_iter(ix).collect::<Vec<_>>());
        } else {
            panic!("Expected cell: {:?}", output)
        }
    }
}

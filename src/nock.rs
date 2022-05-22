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

// if  (   thing  ==     0    )    then  foo  else    bar
// [   [   thing  .      0    ]=   ]?{   foo   }{      bar  }
// dup dup $thing swap lit(0) eq? if[+2] $foo else[+1] bar (fi)
//                                   \----------------^
//                                                \------------^
//
// (cons (a + 1) (2 * b) )
// [     $a+1   .   $2*b   ]
// dup   $a+1 swap  $2*b  cons
//
//
// let  a  1   ...
// [     a=1    .]
// dup  lit(1)  let

// throw
// !!
// get(0)

// let a  (|| printf("hi")) ; a()
// [     a=...             .] a()
// dup lit(some code)    pin  run(2)
//                             0b10   // /[2 a b] => a
//                        nock([a, stuff], a)

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
    const CONS: Elem = 0x8ccc_cccc;
    const SNOC: Elem = 0x8ccc_dddd;
    let mut stack: Vec<Elem> = vec![];

    fn lose_in(heap: &mut Store, stack: &mut Vec<Elem>, subj: Elem){
        if ![CONS,SNOC].contains(&subj) { return }
        let mut deep = 1u32;
        while deep > 0 {
            if ![CONS,SNOC].contains(&stack.pop().expect("Cons-free underflow")) {
                deep += 1
            }
            else {
                deep -= 1;
                heap.lose(subj); //TODO test
            }
        }
    }
    // noun type with shared pointer to the heap
    //  with some kind of miutable .. bullshit, presumanyly a ref sol... tuple of here's heap, here's pointer int oth eheapl..
    // the important point... thin pointer in memory when you're not looking at it... true ither way.. rust want to be safe...
    // involves a lot of bits all the time if you do the obvious thing...
    // build it that way, then build an alternate unsafe, global, notice it doesn't make a difference... (sane plan)
    // that's the sane plan / ok so what are *you(* gonna do
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
            _ => unreachable!()
        };
        let cel = match cdr.into() {
            Noun::Cel(ix) => heap.cons(car,ix).expect("Bad index"),
            _ => heap.pair(car, cdr),
        };
        // println!("pop={} subj={}, car={} cdr={} -> cel={:?} {}",
        //          pop, subj, car, cdr, cel, Into::<Elem>::into(Noun::Cel(cel)));
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
            let mut lose = |subj| { lose_in(heap, &mut stack, subj) };
            match inst {
                Op::Dup => { do_cons(); stack.push(subj) },
                Op::Exch => {
                    do_cons(); //TODO C 1 C 2 3           // oH: also goes in perf epic, whole cons mechanic should get reverted and stuck in a perf epic
                    mem::swap(&mut subj, stack.last_mut().unwrap())
                },
                Op::Pin => { stack.push(subj); subj = CONS },
                Op::Cons => { stack.push(subj); subj = SNOC },
                //
                Op::Fwd(ofs) => {code.drain(..ofs);},
                Op::If(ofs) => {
                    // some negations going on here: if subj, then execute next instruction, else skip ofs
                    if !as_bool(subj) { code.drain(..ofs); }
                    subj = stack.pop().expect("Underflow: no stack after if"); // bool precludes cons
                },
                //                                  [ subject . code ]*
                Op::Nok => {
                    retn.push(code); code = compile(heap,subj);
                    subj = stack.pop().expect("Underflow: no subject for eval");
                }
                Op::Cel => {
                    lose(subj); //TODO drop shenanigans
                    subj = loobean(is_cell(subj) || subj == CONS || subj == SNOC)
                },
                Op::Inc => { assert!(is_atom(subj)); subj += 1 }, //TODO indirect
                Op::Eql => {
                    do_cons();
                    let to = stack.pop().expect("Underflow: no two values to compare");
                    reify_cons(heap, &mut stack, &mut subj);
                    subj = loobean(subj == to || unify(heap, subj, to))
                }
                Op::Lit(e) => { lose(subj); subj = e}
                Op::Get(ax) => {
                    do_cons(); //TODO or navigate them properly
                    //TODO lose rest
                    subj = cdadr(heap, subj, ax).expect("Axed atom")
                }
                Op::Run(ax) => {
                    do_cons(); //TODO or navigate them properly
                    let call = cdadr(heap, subj, ax).expect("Axed atom"); //TODO cons (this is part of issue #17 extract cons onto branch)
                    retn.push(code); code = compile(heap,call);
                }
                Op::Hint => {
                    do_cons();
                    println!("hint: {:?}", Noun::from(subj));
                    //TODO implement hints
                    subj = stack.pop().expect("Underflow: no stack after hint");
                }
                Op::Thin => {
                    todo!("Pop hint-stack");
                }
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

// L: story1: build obvious version of unify-as-isEqual - take two arguments and compare them without leaking them. No perfenhancements, no memory freeing.
// deep unification- two things, tails match, rewrite tail to point to the other, can't drop the whole thing because someone else is hanging onto the whole thing, but parts of them need to be deduplicated.
// TODO: what dedupe guarantees does the unify operator actually provide?
// TODO: separate ewpic for accurate vs performant nock interpreter.
// L: unify story2: what to do about hash memoization (in urbit, check for equlity is amortized fast,
// q1: want to do hashing? -> story: create contrived test case that without hashing is too slow. Is it a test case we are about?
// q2: the name unify - the pair of test cases for that- create either two copies of a large byte buffer and unify them and assert
// that the total memory in use is less than 2 of that byte buffer... or create 2 copies of a deeply nested tree structure and assert that the pointer you get back out is... aaa, nvm
//  a way this can go wrong- it frees something that something else is pointing into because you messed up reference counts- this should be a test for unify.
// 3rd consideration- false positives, false negatives? subcomponent- pointer to compiled code is different from just raw code? Or ...
//  data structure optimization should not-
// need optimized equality algos for things represented in clever ways (that aren't built yet) - i.e. bit streams -
// - that are represented as sequence of bytes with a message on the pointer saying- only go one at a time - want to add that. 32 bitwise comaprison for byte streams, not byte by byte plz.
// leave a t
// TODO detect when things are oif the same optimized representation type and detect when it has an equality special case
// TODO add bit streams - and then need equality for them
// TODO heap: &mut Store, actually unify
fn unify(heap: &Store, a: Elem, b: Elem) -> bool {
    //TODO lose a, b
    if a == b { return true }
    // use Noun::*;
    // match (a.into(),b.into()) {
    //     //TODO test any of this
    //     // (Ind(a), Ind(b)) => {
    //     //     if len(a) != len(b) { return false }
    //     //     for (ax,bx) in iter::zip(a.iter(), b.iter()) {
    //     //         if ax != bx { return false }
    //     //     }
    //     //     return true
    //     // },
    //     // (Cel(a), Cel(b)) => {
    //     //     if len(a) != len(b) { return false }
    //     //     for (ax,bx) in iter::zip(a.iter(), b.iter()) {
    //     //         if !unify(ax, bx) { return false }
    //     //     }
    //     //     return true
    //     // },
    //     _ =>
            false
    // }
}


fn cdadr(mem: &Store, mut e: Elem, mut ax: Axis) -> Option<Elem> {
    assert!(ax != 0);
    //TODO indirect // oH: part of correctness, will take a while to hit
    // takes atom, uses it as a binary address. of most 31 bits. more -> pointer. right now, implementation takes the Elem
    // if index gets too big then it arguabl ... is a number and the
    // if your bigint overflows to be stored ont he heap, you need to be able to read that back out of the heap to use it as an address
    //  dereference heap level pointer to get a nock level pointer
    // ... indirect... least sig bit order... ?
    // traverse list from most to least significant chunk of 32b, then call accent and repeateedly call zbody. still under 10 lines
    // a thing ... that is stupider than xbody ... traverse bigint as a seq of 31bit chunks, or 16bit chunks...
    // worf of ... long thing... 17th bit to 1 ... ? same as ? sequewnce of? since 17 it values that you traversing using... ?
    // ax (access) head... remainian gbits... stupid because you need to repeatedly call function to rederive that 18th but
    // is the most signigifcant bit and you ned to throw away... but also this whole thing is an edge case... and rewriting fewer things...

    let mut a = Noun::from(e);                                    //  0b10  0b0000.0000.0000.0010
    for bit in (0..32usize).into_iter().rev()
                      .skip_while(|bit| 0 == ax & 1<<bit).skip(1) {
        // dbg!(&a, ax, bit, ax & 1<<bit);
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

fn heap() -> &'static mut Store {Box::leak(Box::new(Store::new()))}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn inc(){
        use Op::*;
        assert_eq!(3, nybble(heap(), 2, &[Inc]));
    }

    #[test]
    #[should_panic(expected = "is_atom")]
    fn inc_cel(){
        use Op::*;
        nybble(heap(), 0, &[Dup, Cons, Inc]);
    }

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
        assert_eq!(Noun::Cel(Index(1,0,0)),                                                //[0 [1 2]]   vs
            Noun::from(nybble(store, 1, &[Dup, Inc, Cons, Dup, Lit(0), Pin]))
        );
        // println!("{:?}",Elems(&store));                    [    +    ]     [      0     .]          .] := . ++ ]
    }

    #[test]
    #[should_panic(expected = "Axed atom")]
    fn axe_atom(){
        use Op::*;
        nybble(heap(), 1, &[Get(123)]);
    }

    #[test]
    fn axe(){
        use Op::*;                              //1         1 1   1 2  12  12 12 [12 12]  1
        assert_eq!(1, nybble(heap(), 1, &[Dup, Inc, Cons, Dup, Cons, Get(6)]));          // 0b110
        assert_eq!(2, nybble(heap(), 1, &[Dup, Inc, Cons, Dup, Cons, Get(2), Get(3)]));  // 0b10 0b11
    }                                                   // [    +     ]    [    ]      /2       /3

    #[test]
    fn if_else(){
        use Op::*;
        const CRASH: Op = Get(0);
        assert_eq!(2, nybble(heap(), 0, &[Fwd(1), CRASH, Lit(2)]));
        assert_eq!(3, nybble(heap(), 1, &[Dup, If(1), CRASH, Lit(3)]));
        assert_eq!(44, nybble(heap(), 1, &[Dup, If(2), Lit(99), Fwd(1), Lit(44)]));
        assert_eq!(99, nybble(heap(), 0, &[Dup, If(2), Lit(99), Fwd(1), Lit(44)]));
    }

    #[test]
    #[should_panic(expected = "Underflow")]
    fn if_underflow(){
        use Op::*;
        nybble(heap(), 1, &[If(0)]);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn if_far(){
        use Op::*;
        nybble(heap(), 1, &[Fwd(10)]);
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

    fn program<'a>(a: &mut Store, b: impl Copy + IntoIterator<Item=&'a Op>) -> Elem {
        Noun::Ind(
            a.buffer(b.into_iter().map(|x|x.enc()).collect::<Vec<_>>().flat())
        ).into()
    }
    #[test]
    fn exec(){
        use Op::*;
        let store = heap();
        // println!("{:?}",Elems(&store));       [    1      +   +    +    .]
        let prog = program(store,&[Dup, Lit(1), Inc, Inc, Inc, Pin]);
        // println!("{:?}", Listerator(store.get_iter(match prog.into() { Noun::Ind(x) => x, _=> panic!()})));
        let output = nybble(store, prog, &[Run(1)]).into();
        if let Noun::Cel(ix) = output {
            assert_eq!(vec![4, prog], store.get_iter(ix).collect::<Vec<_>>());
        } else {
            panic!("Expected cell: {:?}", output)
        }
    }

    #[test]
    fn decrement(){
        use Op::*;
        let store = heap();
        let prog = program(store,&[                                                 //   /2    /6    /7
            Dup,                                                 //  [                          //  [acc=0 dec in=10]
              Dup, Get(2), Inc, Exch, Get(7), Eql,               //     [ /2 + . /7 ]=
            If(2), Get(2),                                       //  ]?{ /2
            Fwd(7),                                              //  }{
              Dup, Get(2), Inc, Exch, Get(3), Cons, Run(6)       //    [ /2 + . /3 ] !6
        ]);                                                      //  }              dec(args[0]++, ...args[1...])
        assert_eq!(9, nybble(store, 10, &[Dup, Lit(prog), Pin, Dup, Lit(0), Pin, Run(6)]));
    }                                                  //  [    (dec)    .]     [     0     .]      !6
                                     //        10                       [dec 10]      [0 dec 10]    dec(acc=0,dec,in=10)
                                     //                                               [/2 /6 /7]    6 = 0b110 "args[1]"

    #[test]
    fn reclaims_memory(){
        use Op::*;
        let store = heap();
        let original_used = store.used_bytes();
        assert_eq!(0, nybble(store, 1, &[Dup, Cons, Lit(0)]));
        assert_eq!(original_used, store.used_bytes());
    }
}
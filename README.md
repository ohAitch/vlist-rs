# Running

WORK IN PROGRESS!

(0. `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)  
1\. `cargo test`

# Motivation

Okay so, as per https://github.com/urbit/urbit/blob/master/doc/spec/u3.md
every cell is stored as, effectively,

```c
struct _u3a_boxed_cell {
    c3_w  siz_w;  // size of this box (6)
    c3_w  use_w;  // reference count
    struct u3a_cell {
        c3_w    mug_w;  // hash
        u3_noun hed;    // car
        u3_noun tal;    // cdr
    }
    c3_w  siz_w;  // size of this box (still 6)
}
```

Assuming cells are something like 80% of the loom, this means at any given point a quarter of nonfree memory consists of the bytes `0x0000_0006` repeated over and over and over  
which has at times struck me as… inelegant

(use_w is almost always 1, mug_w is frequently uninitialized at 0, but there are certainly merits in including them, if perhaps not once per byte of a "tape" linked-list or usually-nybble of nock formula. cdr - we'll get to that)

One simple way to reclaim those `siz` bytes is to store them in the pointer: e.g. devote alternating 4kb pages to cells and noncells, such that a bit test can discriminate them. (You can of course still _have_ indirect atoms and other structures >4kb, they just have to begin on an odd-numbered page). This produces a physical memory(cache etc) savings of 25%, and about breaks even on virtual memory(better the closer you are to 50% of your boxes being cells), at the cost of some fragmentation(and 25% more address use if you have _only_ cells or noncells)

A more flexible, but slower, scheme is to use a bitvector of type tags - even a full page granularity of 2^19 bits is only 65kb, fitting easily into L2 cache(at the cost of whatever other things you might on the margin want in L2 cache).

# The pesky cdr field

Of course we are not constrained to only the type tags currently present in u3 (cell vs indirect atom). An ancient idea in the lisp world is https://en.wikipedia.org/wiki/CDR_coding
Where you store `[A B C D]` as a linear array, despite its logical structure of `[A [B [C D]]]`

One can imagine a third "long cell" type, `struct u3a_quad { u3_noun car; u3_noun cadr; u3_noun caddr; u3_noun cdddr}`, for storing tuples of 3 or 4 elements. (Aligned so that a pointer to its `cadr` is distinguishable from a pointer to is `car`, respectively.) This introduces overhead when only 3 elements are present, though not more than if you were to store them as a pair linking to a second pair - but what is "overhead"? The `use` and `mug` have been swept under the rug.  
A simple scheme is to add a header for `use_dot; use_cdr; use_cddr; mug_dot; mug_cdr; mug_cddr`, which rolls back the space savings to only the `cdr` pointers. The three `use` can also be consolidated to a total "references anywhere inside this structure" count, at the cost of occasionally keeping whole quad-cells whose only live data is a `[caddr cdddr]` at the end.  
As for `mug`, caching only `mug_cddr` and forbidding placing quad-cell `dot`/`cdr` pointers in another quad-cell's `car` or `cadr` positions would maintain the constant-time bound. (You can, after all, always allocate a regular cell instead.)

# Vlists

This repository implements an elaboration on the "quad cell" scheme, outlined in https://cl-pdx.com/static/techlists.pdf  
The core conceit is to store a list `~[1 2 3 4 5 6 7 8 9]` as an exponentially flatter sequence

```c
a: {(-:! +<:!) +>-:1 +>+<:2 +>+>-:3 +>+>+<:4 +>+>+>-:5 +>+>+>+<:&b}
b: {-:6 +<:7 +>-:8 +>+:&c}
c: {-:9 +:~}
```

That is, whenever you find yourself `cons`ing onto the middle of a list segment, insert your car as the previous element(if unoccupied) and return a pointer to that; if you `cons` onto an entire list segment, allocate a new segment twice as big - about the current length of the entire list. This uses at most twice as many words as list elements(compare the original linked list scheme which _always_ uses twice as many words), averaging `1.5x`. Placing a bound of 1 page on segment size restricts the maximum absolute overhead to that one page. And now whenever you have a list with a thousand elemends, they're mostly sequential in memory - operations like `lent` and `snag` admit `log N` jets, and unoptimized `cdr` in general-purpose code leaves the cache line prefetcher much happier.

## Metadata

Reference counts are kept in a per-page parallel array of `u16` (overflow scheme tbd for adversarial inputs), figuring that the locality savings for the simple case of read-only access of outer-road data(e.g. library code) outweigh the cache misses on updating them.  
Further work could include implementing "immutable bean" counting in the bytecode compiler to further reduce reference count thrashing, see https://arxiv.org/pdf/1908.05647.pdf

`mug` is not presently implemented, but could work in a manner similar to `rc`.

# Bytes?

In the canonical urbit allocator, `(trip 'abcdefghijklmnop')` producing `"abcdefghijklmnop"` takes a 16-byte indirect atom value(+16 byte overhead), and converts it into a 384-byte linked list value, either a 12x or 24x increase. Under the proposed scheme(assuming small-but-indirect atoms are stuffed into the corresponsing power-of-two segment slabs and pointer-tagged), the 16 byte value(+ 2 byte reference count) will as a linked list occupy ~64 bytes - we've gotten rid of the `cdr`, but the `car` still stores each character as a 31-byte direct atom.  
You could instead imagine _yet another_ pointer tag for "byte vector, but pretend it's a linked list" - `car` is "read the character pointer to a direct atom", `cdr` is "increment pointer, and if it ends up too round replace with `~`", `is_atom` is false and `successor` is crash, technically that's all you need for nock. This suggests a broader field of data-encoding-jets, to match nock execution jets - but this "readme" is getting long as is :)

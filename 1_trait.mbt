///|
pub(open) trait Data: Add + Mul {
  from_double(Double) -> Self
  from_int(Int) -> Self = _
}

///|
impl Data with from_int(i) {
  Data::from_double(i.to_double())
}

///|
pub impl Data for Double with from_double(d) {
  d
}

// ------------------------------------
// |            impl check            |
// ------------------------------------

///|
/// for eval
pub impl Data for Double

///|
/// for forward diff
pub impl[A : Data] Data for Dual[A]

///|
/// for compile
pub impl Data for Atom

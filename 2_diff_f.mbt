// 前向模式微分

///|
struct Dual[A] {
  val : A
  tan : A
} derive(Show)

///|
pub impl[A : Add] Add for Dual[A] with add(a, b) {
  { val: a.val + b.val, tan: a.tan + b.tan }
}

///|
pub impl[A : Add + Mul] Mul for Dual[A] with mul(a, b) {
  { val: a.val * b.val, tan: a.tan * b.val + a.val * b.tan }
}

///|
pub impl[A : Data] Data for Dual[A] with from_double(d) {
  { val: A::from_double(d), tan: A::from_double(0) }
}

///|
pub fn[A : Data] jvp(
  f : (Dual[A]) -> Dual[A],
  val : A,
  tan? : A = A::from_double(1.0),
) -> Dual[A] {
  f({ val, tan })
}

///|
pub fn[A : Data] diff(f : (Dual[A]) -> Dual[A]) -> (A) -> A {
  val => jvp(f, val).tan
}

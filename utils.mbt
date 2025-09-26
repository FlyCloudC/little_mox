///|
fn[A, B] Array::rev_map(self : Array[A], f : (A) -> B) -> Array[B] {
  self.rev_iter().map(f).to_array()
}

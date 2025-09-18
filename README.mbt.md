# FlyCloudC/little_mox

This is a library that demonstrates the use of generics in automatic differentiation. It is similar to the Python JAX library.

## Example

Let's define a function `f` that takes an `A` and returns an `A` for all `A : Data`.

```mbt
fn[A : Data] f(x : A) -> A {
  x * (x + A::from_int(3))
}
```

To compute the derivative of `f`, use `diff`.

```mbt
test "diff" {
  let f_d = f |> diff
  inspect(f_d(2.0), content="7")
}
```

Note that `diff` takes a function that takes a generic type `A : Data` and returns a value of the same type. So we can compute the derivative of the derivative of `f` (i.e. the second derivative) by using `diff(diff(f))`.

```mbt
test "diff then diff" {
  let f_d_d = f |> diff |> diff
  inspect(f_d_d(2.0), content="2")
}
```

The type `Func` represents a compiled function. It can be considered as a list of equations.

```mbt
test "compile" {
  let f_c : Func = f |> Func::compile_1
  assert_eq(f_c.run_1(2.0), f(2.0))
  inspect(
    f_c,
    content=(
      #|p0 => {
      #|  x0 = p0 + 3
      #|  x1 = p0 * x0
      #|  x1
      #|}
    ),
  )
}
```

We can also compile an derivatived function.

```mbt
let f_d_c : Func = f |> diff |> Func::compile_1

test "diff then compile" {
  inspect(
    f_d_c,
    content=(
      #|p0 => {
      #|  x0 = p0 + 3
      #|  x1 = p0 * x0
      #|  x2 = x0 + p0
      #|  x2
      #|}
    ),
  )
}
```

And derivative an compiled function, because `f_d_c.run_1(_)` has type `(A) -> A` for all `A : Data`.

```mbt
let f_d_c_d_c : Func = f_d_c.run_1(_) |> diff |> Func::compile_1

test "diff then compile then diff then compile" {
  inspect(
    f_d_c_d_c,
    content=(
      #|p0 => {
      #|  x0 = p0 + 3
      #|  x1 = p0 * x0
      #|  x2 = x0 + p0
      #|  x3 = x0 + p0
      #|  2
      #|}
    ),
  )
}
```

One last thing is simplifying the equations list.

```mbt
test "simplify" {
  inspect(
    f_d_c.simplify(),
    content=(
      #|p0 => {
      #|  x0 = p0 + 3
      #|  x1 = x0 + p0
      #|  x1
      #|}
    ),
  )
  inspect(f_d_c_d_c.simplify(), content="p0 => 2")
}
```

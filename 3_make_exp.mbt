// ------------------------------------
// |               type               |
// ------------------------------------

///|
pub enum Atom {
  Parm(Int)
  Var(Int)
  Const(Double)
}

///|
enum Exp {
  Add(Atom, Atom)
  Mul(Atom, Atom)
}

///|
pub struct Func {
  params_num : Int
  equtions : Array[Exp]
  ret : Atom
}

// ------------------------------------
// |              effect              |
// ------------------------------------

///|
let current_exps : Ref[Array[Exp]] = { val: [] }

///|
fn[A] with_exps(f : (Array[Exp]) -> A) -> A {
  let exps = []
  let a = current_exps.protect(exps, () => f(exps))
  a
}

///|
fn push_exp(exp : Exp) -> Atom {
  let exps = current_exps.val
  exps.push(exp)
  Var(exps.length() - 1)
}

// ------------------------------------
// |            impl Data             |
// ------------------------------------

///|
pub impl Add for Atom with add(a, b) {
  match (a, b) {
    (Const(a), Const(b)) => Const(a + b)
    (Const(0), x) | (x, Const(0)) => x
    _ => push_exp(Add(a, b))
  }
}

///|
pub impl Mul for Atom with mul(a, b) {
  match (a, b) {
    (Const(a), Const(b)) => Const(a * b)
    (Const(0), _) | (_, Const(0)) => Const(0)
    (Const(1), x) | (x, Const(1)) => x
    _ => push_exp(Mul(a, b))
  }
}

///|
pub impl Data for Atom with from_double(d) {
  Const(d)
}

// ------------------------------------
// |               func               |
// ------------------------------------

///|
fn Exp::map_atom(self : Exp, f : (Atom) -> Atom) -> Exp {
  match self {
    Add(a, b) => Add(f(a), f(b))
    Mul(a, b) => Mul(f(a), f(b))
  }
}

///|
pub fn Func::compile(f : (FixedArray[Atom]) -> Atom, params_num : Int) -> Func {
  let args = FixedArray::makei(params_num, i => Parm(i))
  with_exps(equtions => Func::{ params_num, equtions, ret: f(args) })
}

///|
pub fn Func::simplify(self : Self) -> Self {
  let rev_equtions = []
  let reachable_var_ids = @priority_queue.new()
  let new_vars = FixedArray::make(self.equtions.length(), None)
  let mut next_id = -1
  let to_new_var = a => {
    match a {
      Const(_) | Parm(_) => a
      Var(x) =>
        match new_vars[x] {
          Some(x) => x
          None => {
            reachable_var_ids.push(x)
            let new_var = Var(next_id)
            next_id -= 1
            new_vars[x] = Some(new_var)
            new_var
          }
        }
    }
  }
  let ret = to_new_var(self.ret)
  while reachable_var_ids.pop() is Some(i) {
    rev_equtions.push(self.equtions[i].map_atom(to_new_var))
  }
  let add_length = a => {
    match a {
      Const(_) | Parm(_) => a
      Var(i) => Var(i + rev_equtions.length())
    }
  }
  {
    params_num: self.params_num,
    equtions: rev_equtions.rev_map(expr => expr.map_atom(add_length)),
    ret: add_length(ret),
  }
}

///|
pub fn[A : Data] Func::run(self : Self, args : FixedArray[A]) -> A {
  let vals = []
  fn eval_atom(a : Atom) -> A {
    match a {
      Var(i) => vals[i]
      Parm(i) => args[i]
      Const(d) => A::from_double(d)
    }
  }

  for eq in self.equtions {
    vals.push(
      match eq {
        Add(a, b) => A::add(eval_atom(a), eval_atom(b))
        Mul(a, b) => A::mul(eval_atom(a), eval_atom(b))
      },
    )
  }
  eval_atom(self.ret)
}

///|
pub fn Func::compile_1(f : (Atom) -> Atom) -> Func {
  Func::compile(x => f(x[0]), 1)
}

///|
pub fn[A : Data] Func::run_1(self : Self, arg : A) -> A {
  self.run([arg])
}

// ------------------------------------
// |            impl Show             |
// ------------------------------------

///|
pub impl Show for Atom with output(self, logger) {
  match self {
    Parm(i) => logger..write_char('p').write_object(i)
    Var(i) => logger..write_char('x').write_object(i)
    Const(d) => logger.write_object(d)
  }
}

///|
pub impl Show for Exp with output(self, logger) {
  match self {
    Add(a, b) => logger..write_object(a)..write_string(" + ").write_object(b)
    Mul(a, b) => logger..write_object(a)..write_string(" * ").write_object(b)
  }
}

///|
pub impl Show for Func with output(self, logger) {
  match self.params_num {
    0 => logger.write_string("()")
    1 => logger.write_object(Parm(0))
    _ => {
      logger.write_char('(')
      logger.write_object(Parm(0))
      for i in 1..<self.params_num {
        logger.write_object(Parm(i))
      }
      logger.write_char(')')
    }
  }
  logger.write_string(" => ")
  match self.equtions {
    [] => logger.write_object(self.ret)
    eqs => {
      logger.write_string("{")
      for i, e in eqs {
        logger
        ..write_string("\n  ")
        ..write_object(Var(i))
        ..write_string(" = ")
        .write_object(e)
      }
      logger..write_string("\n  ")..write_object(self.ret).write_string("\n}")
    }
  }
}

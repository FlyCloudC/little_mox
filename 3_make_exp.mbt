///|
pub enum Atom {
  Parm(Int)
  Var(Int)
  Data(Double)
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

///|
let current_exps : Ref[Array[Exp]] = { val: [] }

///|
pub impl Add for Atom with add(a, b) {
  match (a, b) {
    (Data(a), Data(b)) => Data(a + b)
    (Data(0), x) | (x, Data(0)) => x
    _ => {
      let exps = current_exps.val
      defer exps.push(Add(a, b))
      Var(exps.length())
    }
  }
}

///|
pub impl Mul for Atom with mul(a, b) {
  match (a, b) {
    (Data(a), Data(b)) => Data(a * b)
    (Data(0), _) | (_, Data(0)) => Data(0)
    (Data(1), x) | (x, Data(1)) => x
    _ => {
      let exps = current_exps.val
      defer exps.push(Mul(a, b))
      Var(exps.length())
    }
  }
}

///|
pub impl Data for Atom with from_double(d) {
  Data(d)
}

///|
pub fn Func::compile(f : (FixedArray[Atom]) -> Atom, params_num : Int) -> Func {
  let args = FixedArray::makei(params_num, Parm(_))
  let ret = f(args)
  let equtions = current_exps.val
  current_exps.val = []
  Func::{ params_num, equtions, ret }
}

///|
pub fn Func::simplify(self : Self) -> Self {
  let res = []
  let pq = @priority_queue.new()
  let id = FixedArray::make(self.equtions.length(), None)
  let mut cur_id = 0
  let eval_atom = a => match a {
    Data(_) | Parm(_) => a
    Var(x) =>
      match id[x] {
        Some(x) => x
        None => {
          pq.push(x)
          cur_id -= 1
          let r = Var(cur_id)
          id[x] = Some(r)
          r
        }
      }
  }
  let ret = eval_atom(self.ret)
  while pq.pop() is Some(i) {
    res.push(
      match self.equtions[i] {
        Add(a, b) => Add(eval_atom(a), eval_atom(b))
        Mul(a, b) => Mul(eval_atom(a), eval_atom(b))
      },
    )
  }
  res.rev_inplace()
  let add_length = a => match a {
    Data(_) | Parm(_) => a
    Var(i) => Var(i + res.length())
  }
  {
    params_num: self.params_num,
    equtions: res.map(exp => match exp {
      Add(a, b) => Add(add_length(a), add_length(b))
      Mul(a, b) => Mul(add_length(a), add_length(b))
    }),
    ret: add_length(ret),
  }
}

///|
pub fn[A : Data] Func::run(self : Self, args : FixedArray[A]) -> A {
  let res = []
  fn eval_atom(a : Atom) -> A {
    match a {
      Var(i) => res[i]
      Parm(i) => args[i]
      Data(d) => A::from_double(d)
    }
  }

  for eq in self.equtions {
    res.push(
      match eq {
        Add(a, b) => eval_atom(a) + eval_atom(b)
        Mul(a, b) => eval_atom(a) * eval_atom(b)
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
// |               Show               |
// ------------------------------------

///|
pub impl Show for Atom with output(self, logger) {
  match self {
    Parm(i) => logger..write_char('p')..write_object(i)
    Var(i) => logger..write_char('x')..write_object(i)
    Data(d) => logger.write_object(d)
  }
}

///|
pub impl Show for Exp with output(self, logger) {
  match self {
    Add(a, b) => logger..write_object(a)..write_string(" + ")..write_object(b)
    Mul(a, b) => logger..write_object(a)..write_string(" * ")..write_object(b)
  }
}

///|
pub impl Show for Func with output(self, logger) {
  match self.params_num {
    0 => logger.write_string("()")
    1 => logger..write_object(Parm(0))
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
    [] => logger..write_object(self.ret)
    eqs => {
      logger.write_string("{")
      for i, e in eqs {
        logger
        ..write_string("\n  ")
        ..write_object(Var(i))
        ..write_string(" = ")
        ..write_object(e)
      }
      logger..write_string("\n  ")..write_object(self.ret)..write_string("\n}")
    }
  }
}

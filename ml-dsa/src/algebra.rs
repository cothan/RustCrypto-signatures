use core::ops::{Add, Mul, Sub};
use hybrid_array::{typenum::U256, Array};
use sha3::digest::XofReader;

pub type Integer = u32;

/// An element of GF(q).  Although `q` is only 16 bits wide, we use a wider uint type to so that we
/// can defer modular reductions.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct FieldElement(pub Integer);

impl FieldElement {
    pub const Q: u32 = 8380417;
    pub const Q64: u64 = Self::Q as u64;
    const QINV: u64 = 58728449;

    // Constant time (hopefully) small reduce
    fn small_reduce(x: u32) -> u32 {
        let mask = (x > Self::Q) as u32;
        x - (mask * Self::Q)
    }

    // Algorithm 37. Montgomery Reduction
    fn montgomery_mul(a: Self, b: Self) -> Self {
        let a = u64::from(a.0) * u64::from(b.0);
        let t = (u64::from(a as u32) * Self::QINV) as u32;
        let r = (a - u64::from(t) * Self::Q64) >> 32;
        Self(Self::small_reduce(r as u32))
    }
}

impl Add<FieldElement> for FieldElement {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self(Self::small_reduce(self.0 + rhs.0))
    }
}

impl Sub<FieldElement> for FieldElement {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        // Guard against underflow if `rhs` is too large
        Self(Self::small_reduce(self.0 + Self::Q - rhs.0))
    }
}

impl Mul<FieldElement> for FieldElement {
    type Output = FieldElement;

    fn mul(self, rhs: FieldElement) -> FieldElement {
        Self::montgomery_mul(self, rhs)
    }
}

/// An element of the ring `R_q`, i.e., a polynomial over `Z_q` of degree 256
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Polynomial(pub Array<FieldElement, U256>);

impl Add<&Polynomial> for &Polynomial {
    type Output = Polynomial;

    fn add(self, rhs: &Polynomial) -> Polynomial {
        Polynomial(
            self.0
                .iter()
                .zip(rhs.0.iter())
                .map(|(&x, &y)| x + y)
                .collect(),
        )
    }
}

impl Sub<&Polynomial> for &Polynomial {
    type Output = Polynomial;

    fn sub(self, rhs: &Polynomial) -> Polynomial {
        Polynomial(
            self.0
                .iter()
                .zip(rhs.0.iter())
                .map(|(&x, &y)| x - y)
                .collect(),
        )
    }
}

impl Mul<&Polynomial> for FieldElement {
    type Output = Polynomial;

    fn mul(self, rhs: &Polynomial) -> Polynomial {
        Polynomial(rhs.0.iter().map(|&x| self * x).collect())
    }
}

#[cfg(test)]
mod test {}

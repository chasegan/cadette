//! Length units and the document's working unit.
//!
//! The kernel works in a single internal unit (millimeters); UI and import
//! layers convert at the boundary. This seed covers the conversions Phase 0
//! needs and gives later parametric work (`wall = 3mm`) a place to grow.

/// A length unit the UI can present and parse.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LengthUnit {
    Millimeter,
    Centimeter,
    Meter,
    Inch,
}

impl LengthUnit {
    /// Millimeters per one of this unit. Millimeter is the kernel's internal
    /// unit, so these factors convert *into* kernel space.
    pub const fn mm_per_unit(self) -> f64 {
        match self {
            LengthUnit::Millimeter => 1.0,
            LengthUnit::Centimeter => 10.0,
            LengthUnit::Meter => 1000.0,
            LengthUnit::Inch => 25.4,
        }
    }

    /// Convert a value in this unit to internal millimeters.
    pub fn to_mm(self, value: f64) -> f64 {
        value * self.mm_per_unit()
    }

    /// Convert internal millimeters to a value in this unit.
    pub fn from_mm(self, mm: f64) -> f64 {
        mm / self.mm_per_unit()
    }

    /// The short suffix shown in the UI.
    pub const fn suffix(self) -> &'static str {
        match self {
            LengthUnit::Millimeter => "mm",
            LengthUnit::Centimeter => "cm",
            LengthUnit::Meter => "m",
            LengthUnit::Inch => "in",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inch_roundtrips_through_mm() {
        let one_inch_mm = LengthUnit::Inch.to_mm(1.0);
        assert!((one_inch_mm - 25.4).abs() < 1e-9);
        assert!((LengthUnit::Inch.from_mm(one_inch_mm) - 1.0).abs() < 1e-9);
    }
}

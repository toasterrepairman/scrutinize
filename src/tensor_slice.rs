use std::collections::HashMap;

/// Defines how to slice a multi-dimensional tensor for 2D visualization
#[derive(Debug, Clone)]
pub struct SliceSelection {
    /// Total shape of the tensor
    pub shape: Vec<u64>,

    /// Which two dimensions to display (indices into shape)
    /// First is rows (y), second is columns (x)
    pub display_dims: (usize, usize),

    /// Fixed indices for dimensions not being displayed
    /// Maps dimension index -> selected index in that dimension
    pub fixed_indices: HashMap<usize, u64>,
}

impl SliceSelection {
    /// Create a default slice selection for a given tensor shape
    /// For 2D: display as-is
    /// For 3D+: display last two dimensions, fix others at 0
    pub fn default_for_shape(shape: Vec<u64>) -> Self {
        match shape.len() {
            0 | 1 => panic!("Cannot create 2D slice from 0D or 1D tensor"),
            2 => Self {
                display_dims: (0, 1),
                fixed_indices: HashMap::new(),
                shape,
            },
            n => {
                // Display last two dimensions
                let display_dims = (n - 2, n - 1);

                // Fix all other dimensions to index 0
                let mut fixed_indices = HashMap::new();
                for i in 0..(n - 2) {
                    fixed_indices.insert(i, 0);
                }

                Self {
                    shape,
                    display_dims,
                    fixed_indices,
                }
            }
        }
    }

    /// Create a smart default based on tensor name and shape
    /// Strategy: Always display the TWO LARGEST dimensions (resampled into heatmap)
    /// and provide sliders for all smaller dimensions. This ensures:
    /// 1. Maximum data visibility (large dimensions shown)
    /// 2. Minimal slicing needed (small dimensions controlled via sliders)
    /// 3. Intuitive navigation (slice through fewer, smaller layers)
    pub fn smart_default(name: &str, shape: Vec<u64>) -> Self {
        if shape.len() <= 2 {
            return Self::default_for_shape(shape);
        }

        // For 3D+ tensors, ALWAYS display the two largest dimensions
        // and slice through all smaller dimensions

        // Create a list of (dimension_index, dimension_size) and sort by size descending
        let mut dims_with_size: Vec<(usize, u64)> = shape.iter()
            .enumerate()
            .map(|(idx, &size)| (idx, size))
            .collect();
        dims_with_size.sort_by_key(|(_, size)| std::cmp::Reverse(*size));

        // Take the two largest dimensions
        let largest_dim_0 = dims_with_size[0].0;
        let largest_dim_1 = dims_with_size[1].0;

        // Ensure display_dims are ordered by index (for consistency in layout)
        let display_dims = if largest_dim_0 < largest_dim_1 {
            (largest_dim_0, largest_dim_1)
        } else {
            (largest_dim_1, largest_dim_0)
        };

        // Fix all non-displayed dimensions to 0 (these will be the slider controls)
        let mut fixed_indices = HashMap::new();
        for i in 0..shape.len() {
            if i != display_dims.0 && i != display_dims.1 {
                fixed_indices.insert(i, 0);
            }
        }

        Self {
            shape,
            display_dims,
            fixed_indices,
        }
    }

    /// Get the 2D shape of the displayable slice (height, width)
    pub fn slice_shape(&self) -> (u64, u64) {
        (
            self.shape[self.display_dims.0],
            self.shape[self.display_dims.1],
        )
    }

    /// Update which index is selected for a fixed dimension
    pub fn set_fixed_index(&mut self, dim: usize, index: u64) -> Result<(), String> {
        if dim >= self.shape.len() {
            return Err(format!("Dimension {} out of bounds", dim));
        }

        if dim == self.display_dims.0 || dim == self.display_dims.1 {
            return Err(format!("Cannot fix dimension {} - it's being displayed", dim));
        }

        if index >= self.shape[dim] {
            return Err(format!("Index {} out of bounds for dimension {} (size {})",
                index, dim, self.shape[dim]));
        }

        self.fixed_indices.insert(dim, index);
        Ok(())
    }

    /// Calculate the linear offset for a given 2D position in the slice
    /// Returns None if the position is out of bounds
    /// This assumes row-major (C-style) memory layout
    pub fn linear_offset(&self, row: u64, col: u64) -> Option<u64> {
        let (height, width) = self.slice_shape();

        if row >= height || col >= width {
            return None;
        }

        // Build full N-dimensional index
        let mut indices = vec![0u64; self.shape.len()];

        // Set fixed indices
        for (&dim, &idx) in &self.fixed_indices {
            indices[dim] = idx;
        }

        // Set display indices
        indices[self.display_dims.0] = row;
        indices[self.display_dims.1] = col;

        // Calculate linear offset (row-major)
        Some(Self::nd_to_linear(&indices, &self.shape))
    }

    /// Convert N-dimensional indices to linear offset (row-major layout)
    fn nd_to_linear(indices: &[u64], shape: &[u64]) -> u64 {
        let mut offset = 0u64;
        let mut stride = 1u64;

        // Iterate from last dimension to first (row-major)
        for i in (0..shape.len()).rev() {
            offset += indices[i] * stride;
            stride *= shape[i];
        }

        offset
    }

    /// Check if this tensor needs slicing (is 3D or higher)
    pub fn needs_slicing(&self) -> bool {
        self.shape.len() > 2
    }

    /// Get a list of dimensions that can be sliced through
    /// Returns (dim_index, dim_name, dim_size, current_index)
    pub fn sliceable_dimensions(&self) -> Vec<(usize, String, u64, u64)> {
        let mut result = Vec::new();

        for (dim_idx, &size) in self.shape.iter().enumerate() {
            // Skip dimensions being displayed
            if dim_idx == self.display_dims.0 || dim_idx == self.display_dims.1 {
                continue;
            }

            let current_idx = *self.fixed_indices.get(&dim_idx).unwrap_or(&0);
            let name = Self::dimension_name(dim_idx, self.shape.len(), &self.shape);

            result.push((dim_idx, name, size, current_idx));
        }

        result
    }

    /// Generate a human-readable name for a dimension
    fn dimension_name(dim_idx: usize, total_dims: usize, shape: &[u64]) -> String {
        let size = shape[dim_idx];

        // Since we always slice through smaller dimensions now, use descriptive names
        // based on common patterns and position
        let base_name = match (dim_idx, total_dims) {
            // 3D tensors
            (0, 3) => "Layer",
            (1, 3) => "Channel",
            (2, 3) => "Depth",
            // 4D tensors
            (0, 4) => "Batch",
            (1, 4) => "Channel",
            (2, 4) => "Layer",
            (3, 4) => "Head",
            // Generic
            _ => "Dimension",
        };

        format!("{} ({})", base_name, size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_2d_tensor() {
        let slice = SliceSelection::default_for_shape(vec![10, 20]);
        assert_eq!(slice.slice_shape(), (10, 20));
        assert_eq!(slice.linear_offset(0, 0), Some(0));
        assert_eq!(slice.linear_offset(5, 10), Some(5 * 20 + 10));
        assert!(!slice.needs_slicing());
    }

    #[test]
    fn test_3d_tensor() {
        let slice = SliceSelection::default_for_shape(vec![32, 128, 64]);
        assert_eq!(slice.slice_shape(), (128, 64));
        assert!(slice.needs_slicing());

        // At fixed index [0, row, col]
        assert_eq!(slice.linear_offset(0, 0), Some(0));
        assert_eq!(slice.linear_offset(1, 0), Some(64));
        assert_eq!(slice.linear_offset(0, 1), Some(1));
    }

    #[test]
    fn test_4d_tensor() {
        let mut slice = SliceSelection::default_for_shape(vec![8, 32, 128, 64]);

        // Default should fix dims 0 and 1 to 0
        assert_eq!(slice.slice_shape(), (128, 64));

        // Change fixed index for dimension 1
        slice.set_fixed_index(1, 5).unwrap();

        // Should now be at [0, 5, row, col]
        // Offset = 0*32*128*64 + 5*128*64 + row*64 + col
        let expected = 5 * 128 * 64;
        assert_eq!(slice.linear_offset(0, 0), Some(expected));
    }

    #[test]
    fn test_sliceable_dimensions() {
        let slice = SliceSelection::default_for_shape(vec![8, 32, 128, 64]);
        let sliceable = slice.sliceable_dimensions();

        // Should have 2 sliceable dims (0 and 1), since 2 and 3 are being displayed
        assert_eq!(sliceable.len(), 2);
        assert_eq!(sliceable[0].0, 0); // dim index
        assert_eq!(sliceable[0].2, 8);  // dim size
        assert_eq!(sliceable[1].0, 1);
        assert_eq!(sliceable[1].2, 32);
    }

    #[test]
    fn test_smart_default_slices_smallest_dimension() {
        // For a tensor like [2880, 2880, 32], we want to display 2880x2880
        // and slice through the 32 (dimension 2)
        let slice = SliceSelection::smart_default("test", vec![2880, 2880, 32]);

        // Should display the two largest dimensions (0 and 1)
        assert_eq!(slice.display_dims, (0, 1));
        assert_eq!(slice.slice_shape(), (2880, 2880));

        // Should have fixed dimension 2 (the smallest)
        assert_eq!(slice.fixed_indices.len(), 1);
        assert_eq!(slice.fixed_indices.get(&2), Some(&0));

        // The sliceable dimension should be dimension 2 with size 32
        let sliceable = slice.sliceable_dimensions();
        assert_eq!(sliceable.len(), 1);
        assert_eq!(sliceable[0].0, 2); // dimension index
        assert_eq!(sliceable[0].2, 32); // dimension size
    }

    #[test]
    fn test_smart_default_with_small_first_dim() {
        // For a tensor like [32, 4096, 4096], we want to display 4096x4096
        // and slice through the 32 (dimension 0)
        let slice = SliceSelection::smart_default("test", vec![32, 4096, 4096]);

        // Should display dimensions 1 and 2 (the two largest)
        assert_eq!(slice.display_dims, (1, 2));
        assert_eq!(slice.slice_shape(), (4096, 4096));

        // Should have fixed dimension 0 (the smallest)
        assert_eq!(slice.fixed_indices.len(), 1);
        assert_eq!(slice.fixed_indices.get(&0), Some(&0));
    }
}

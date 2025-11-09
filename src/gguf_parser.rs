use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use memmap2::Mmap;
use std::sync::Arc;

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in ASCII
const GGUF_VERSION: u32 = 3;

#[derive(Debug, Clone)]
pub struct GGUFFile {
    pub version: u32,
    pub metadata: GGUFMetadata,
    pub tensors: Vec<TensorInfo>,
    pub file_size: u64,
    mmap: Option<Arc<Mmap>>,  // Memory-mapped file for efficient access
    file_path: Option<std::path::PathBuf>,  // Store file path for lazy mmap initialization
}

#[derive(Debug, Clone)]
pub struct GGUFMetadata {
    values: HashMap<String, MetadataValue>,
}

#[derive(Debug, Clone)]
pub enum MetadataValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    UInt64(u64),
    Int64(i64),
    Float64(f64),
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub dtype: GGMLType,
    pub offset: u64,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
    MXFP4 = 39,
}

impl GGMLType {
    fn from_u32(value: u32) -> Result<Self, String> {
        match value {
            0 => Ok(GGMLType::F32),
            1 => Ok(GGMLType::F16),
            2 => Ok(GGMLType::Q4_0),
            3 => Ok(GGMLType::Q4_1),
            6 => Ok(GGMLType::Q5_0),
            7 => Ok(GGMLType::Q5_1),
            8 => Ok(GGMLType::Q8_0),
            9 => Ok(GGMLType::Q8_1),
            10 => Ok(GGMLType::Q2_K),
            11 => Ok(GGMLType::Q3_K),
            12 => Ok(GGMLType::Q4_K),
            13 => Ok(GGMLType::Q5_K),
            14 => Ok(GGMLType::Q6_K),
            15 => Ok(GGMLType::Q8_K),
            16 => Ok(GGMLType::I8),
            17 => Ok(GGMLType::I16),
            18 => Ok(GGMLType::I32),
            39 => Ok(GGMLType::MXFP4),
            _ => Err(format!("Unknown GGML type: {}", value)),
        }
    }

    pub fn bits_per_weight(&self) -> f32 {
        match self {
            GGMLType::F32 | GGMLType::I32 => 32.0,
            GGMLType::F16 | GGMLType::I16 => 16.0,
            GGMLType::Q4_0 | GGMLType::Q4_1 => 4.5,
            GGMLType::Q5_0 | GGMLType::Q5_1 => 5.5,
            GGMLType::Q8_0 | GGMLType::Q8_1 | GGMLType::I8 => 8.5,
            GGMLType::Q2_K => 2.5625,
            GGMLType::Q3_K => 3.4375,
            GGMLType::Q4_K => 4.5,
            GGMLType::Q5_K => 5.5,
            GGMLType::Q6_K => 6.5625,
            GGMLType::Q8_K => 8.5,
            GGMLType::MXFP4 => 4.0,  // Microscaling FP4: 4 bits per weight
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            GGMLType::F32 => "F32",
            GGMLType::F16 => "F16",
            GGMLType::Q4_0 => "Q4_0",
            GGMLType::Q4_1 => "Q4_1",
            GGMLType::Q5_0 => "Q5_0",
            GGMLType::Q5_1 => "Q5_1",
            GGMLType::Q8_0 => "Q8_0",
            GGMLType::Q8_1 => "Q8_1",
            GGMLType::Q2_K => "Q2_K",
            GGMLType::Q3_K => "Q3_K",
            GGMLType::Q4_K => "Q4_K",
            GGMLType::Q5_K => "Q5_K",
            GGMLType::Q6_K => "Q6_K",
            GGMLType::Q8_K => "Q8_K",
            GGMLType::I8 => "I8",
            GGMLType::I16 => "I16",
            GGMLType::I32 => "I32",
            GGMLType::MXFP4 => "MXFP4",
        }
    }
}

impl GGUFFile {
    pub fn load(path: &Path) -> Result<Self, String> {
        let mut file = File::open(path)
            .map_err(|e| format!("Failed to open file: {}", e))?;

        let file_size = file.metadata()
            .map_err(|e| format!("Failed to get file size: {}", e))?
            .len();

        // Read magic number
        let magic = read_u32(&mut file)?;
        if magic != GGUF_MAGIC {
            return Err(format!("Invalid GGUF magic number: 0x{:08X}", magic));
        }

        // Read version
        let version = read_u32(&mut file)?;
        if version < 2 || version > 3 {
            return Err(format!("Unsupported GGUF version: {}", version));
        }

        // Read tensor count
        let tensor_count = read_u64(&mut file)?;

        // Read metadata count
        let metadata_count = read_u64(&mut file)?;

        // Read metadata
        let mut metadata_values = HashMap::new();
        for _ in 0..metadata_count {
            let (key, value) = read_metadata_kv(&mut file)?;
            metadata_values.insert(key, value);
        }

        // Read tensor info
        let mut tensors = Vec::new();
        for _ in 0..tensor_count {
            let tensor_info = read_tensor_info(&mut file)?;
            tensors.push(tensor_info);
        }

        // Calculate alignment
        let alignment = metadata_values
            .get("general.alignment")
            .and_then(|v| {
                if let MetadataValue::UInt32(a) = v {
                    Some(*a as u64)
                } else {
                    None
                }
            })
            .unwrap_or(32);

        // Calculate tensor data offset
        let current_pos = file.stream_position()
            .map_err(|e| format!("Failed to get position: {}", e))?;
        let tensor_data_offset = ((current_pos + alignment - 1) / alignment) * alignment;

        // Update tensor offsets to be absolute
        for tensor in &mut tensors {
            tensor.offset += tensor_data_offset;
        }

        Ok(GGUFFile {
            version,
            metadata: GGUFMetadata {
                values: metadata_values,
            },
            tensors,
            file_size,
            mmap: None,  // Will be initialized lazily when needed
            file_path: Some(path.to_path_buf()),
        })
    }

    /// Initialize memory mapping for efficient tensor access
    pub fn init_mmap(&mut self) -> Result<(), String> {
        if self.mmap.is_none() {
            let path = self.file_path.as_ref()
                .ok_or("File path not available for memory mapping")?;

            let file = File::open(path)
                .map_err(|e| format!("Failed to open file for memory mapping: {}", e))?;

            let mmap = unsafe { Mmap::map(&file) }
                .map_err(|e| format!("Failed to create memory map: {}", e))?;

            self.mmap = Some(Arc::new(mmap));
        }
        Ok(())
    }

    /// Get memory map, initializing if necessary
    pub fn get_mmap(&self) -> Result<Arc<Mmap>, String> {
        if let Some(ref mmap) = self.mmap {
            Ok(mmap.clone())
        } else {
            // Need mutable reference to initialize, but we're in an immutable context
            // This suggests we need to restructure the API
            Err("Memory map not initialized. Call init_mmap() first.".to_string())
        }
    }

    /// Optimized tensor reading that combines all improvements
    /// This is the preferred method for accessing tensor data
    pub fn read_tensor_data_optimized(
        &mut self,
        tensor_name: &str,
        max_elements: usize,
        slice_selection: Option<&crate::tensor_slice::SliceSelection>
    ) -> Result<Vec<f32>, String> {
        // Initialize memory mapping if not already done
        self.init_mmap()?;

        // Find the tensor
        let tensor = self.tensors.iter()
            .find(|t| t.name == tensor_name)
            .ok_or(format!("Tensor '{}' not found", tensor_name))?;

        // Get the memory map
        let mmap = self.get_mmap()?;

        // Use the optimized tensor reading method
        tensor.read_tensor_data_optimized(&*mmap, max_elements, slice_selection)
    }

    /// Get tensor info without reading data
    pub fn get_tensor_info(&self, tensor_name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == tensor_name)
    }

    /// List all tensor names
    pub fn list_tensor_names(&self) -> Vec<&String> {
        self.tensors.iter().map(|t| &t.name).collect()
    }

    /// Get performance statistics for the loaded model
    pub fn get_performance_stats(&self) -> PerformanceStats {
        PerformanceStats {
            total_tensors: self.tensors.len(),
            total_parameters: self.compute_total_parameters(),
            total_memory_bytes: self.compute_memory_requirements(),
            memory_mapping_enabled: self.mmap.is_some(),
            quantized_tensors: self.tensors.iter()
                .filter(|t| matches!(t.dtype, GGMLType::Q4_0 | GGMLType::Q4_1 | GGMLType::Q8_0 | GGMLType::Q2_K | GGMLType::Q3_K | GGMLType::Q4_K | GGMLType::Q5_K | GGMLType::Q6_K | GGMLType::Q8_K))
                .count(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceStats {
    pub total_tensors: usize,
    pub total_parameters: u64,
    pub total_memory_bytes: u64,
    pub memory_mapping_enabled: bool,
    pub quantized_tensors: usize,
}

impl std::fmt::Display for PerformanceStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Performance Statistics:")?;
        writeln!(f, "  Total Tensors: {}", self.total_tensors)?;
        writeln!(f, "  Total Parameters: {:.2}B", self.total_parameters as f64 / 1e9)?;
        writeln!(f, "  Total Memory: {:.2}GB", self.total_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0))?;
        writeln!(f, "  Memory Mapping: {}", if self.memory_mapping_enabled { "Enabled" } else { "Disabled" })?;
        writeln!(f, "  Quantized Tensors: {} ({:.1}%)",
            self.quantized_tensors,
            (self.quantized_tensors as f64 / self.total_tensors as f64) * 100.0
        )?;
        Ok(())
    }
}

impl GGUFFile {
    pub fn compute_total_parameters(&self) -> u64 {
        self.tensors.iter().map(|t| {
            t.dimensions.iter().product::<u64>()
        }).sum()
    }

    pub fn get_vocab_size(&self) -> Option<usize> {
        self.metadata.values.get("tokenizer.ggml.tokens")
            .and_then(|v| {
                if let MetadataValue::Array(tokens) = v {
                    Some(tokens.len())
                } else {
                    None
                }
            })
    }

    pub fn compute_memory_requirements(&self) -> u64 {
        self.tensors.iter().map(|t| t.size_bytes).sum()
    }
}

impl GGUFMetadata {
    pub fn get_string(&self, key: &str) -> Option<String> {
        self.values.get(key).and_then(|v| {
            if let MetadataValue::String(s) = v {
                Some(s.clone())
            } else {
                None
            }
        })
    }

    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.values.get(key).and_then(|v| {
            match v {
                MetadataValue::UInt32(n) => Some(*n),
                MetadataValue::UInt64(n) => Some(*n as u32),
                _ => None,
            }
        })
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.values.get(key).and_then(|v| {
            if let MetadataValue::Float32(f) = v {
                Some(*f)
            } else {
                None
            }
        })
    }

    pub fn get_array(&self, key: &str) -> Option<&Vec<MetadataValue>> {
        self.values.get(key).and_then(|v| {
            if let MetadataValue::Array(arr) = v {
                Some(arr)
            } else {
                None
            }
        })
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.values.keys()
    }
}

// Dequantization functions for different GGML types
mod dequantize {
    use super::GGMLType;
    use half::f16;

    /// Dequantize Q4_0 format: 8 quantized values + 1 fp16 scale
    pub fn q4_0(data: &[u8], count: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(count);
        let chunks = data.chunks_exact(9); // Q4_0 block size

        for chunk in chunks {
            if result.len() >= count {
                break;
            }

            // Read scale (fp16)
            let scale_bytes = [chunk[0], chunk[1]];
            let scale = f16::from_le_bytes(scale_bytes).to_f32();

            // Read 8 quantized values (4 bits each)
            for i in 0..8 {
                if result.len() >= count {
                    break;
                }
                let byte_idx = 2 + i / 2;
                let quantized = (chunk[byte_idx] >> ((i % 2) * 4)) & 0x0F;
                let value = (quantized as f32 - 8.0) * scale;
                result.push(value);
            }
        }

        result
    }

    /// Dequantize Q4_1 format: 8 quants + 1 fp16 min + 1 fp16 scale
    pub fn q4_1(data: &[u8], count: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(count);
        let chunks = data.chunks_exact(10); // Q4_1 block size

        for chunk in chunks {
            if result.len() >= count {
                break;
            }

            // Read min and scale (both fp16)
            let min_bytes = [chunk[0], chunk[1]];
            let scale_bytes = [chunk[2], chunk[3]];
            let min = f16::from_le_bytes(min_bytes).to_f32();
            let scale = f16::from_le_bytes(scale_bytes).to_f32();

            // Read 8 quantized values
            for i in 0..8 {
                if result.len() >= count {
                    break;
                }
                let byte_idx = 4 + i / 2;
                let quantized = (chunk[byte_idx] >> ((i % 2) * 4)) & 0x0F;
                let value = min + (quantized as f32 * scale);
                result.push(value);
            }
        }

        result
    }

    /// Dequantize Q8_0 format: 32 quantized values + 1 fp16 scale
    pub fn q8_0(data: &[u8], count: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(count);
        let chunks = data.chunks_exact(34); // Q8_0 block size

        for chunk in chunks {
            if result.len() >= count {
                break;
            }

            // Read scale (fp16)
            let scale_bytes = [chunk[0], chunk[1]];
            let scale = f16::from_le_bytes(scale_bytes).to_f32();

            // Read 32 quantized values (int8 each)
            for i in 0..32 {
                if result.len() >= count {
                    break;
                }
                let quantized = chunk[2 + i] as i8 as f32;
                let value = quantized * scale;
                result.push(value);
            }
        }

        result
    }

    /// Simplified dequantization for other quantized types
    /// For production use, implement proper dequantization for each type
    pub fn simplified(data: &[u8], count: usize) -> Vec<f32> {
        data.iter()
            .take(count)
            .map(|&b| (b as f32 - 128.0) / 128.0) // Normalize to -1..1
            .collect()
    }

    /// Convert f16 to f32 using half crate
    pub fn f16_to_f32(data: &[u8], count: usize) -> Vec<f32> {
        let chunks = data.chunks_exact(2);
        chunks
            .take(count)
            .map(|chunk| {
                let f16_val = f16::from_le_bytes([chunk[0], chunk[1]]);
                f16_val.to_f32()
            })
            .collect()
    }
}

impl TensorInfo {
    pub fn element_count(&self) -> u64 {
        self.dimensions.iter().product()
    }

    pub fn shape_string(&self) -> String {
        format!("[{}]", self.dimensions.iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", "))
    }

    /// Get the number of bytes per element for this tensor's data type
    fn bytes_per_element(&self) -> u64 {
        match self.dtype {
            GGMLType::F32 | GGMLType::I32 => 4,
            GGMLType::F16 | GGMLType::I16 => 2,
            GGMLType::I8 => 1,
            // For quantized types, approximate
            _ => {
                let element_count = self.element_count();
                if element_count > 0 {
                    (self.size_bytes + element_count - 1) / element_count
                } else {
                    1
                }
            }
        }
    }

    /// Optimized tensor data reading using memory mapping
    /// This is the primary method that should be used for tensor access
    pub fn read_tensor_data_optimized(
        &self,
        mmap: &memmap2::Mmap,
        max_elements: usize,
        slice_selection: Option<&crate::tensor_slice::SliceSelection>
    ) -> Result<Vec<f32>, String> {
        if let Some(slice) = slice_selection {
            self.read_slice_mmap(mmap, slice, max_elements)
        } else {
            self.read_entire_tensor_mmap(mmap, max_elements)
        }
    }

    /// Read entire tensor using memory mapping (no file I/O!)
    fn read_entire_tensor_mmap(
        &self,
        mmap: &memmap2::Mmap,
        max_elements: usize,
    ) -> Result<Vec<f32>, String> {
        let element_count = self.element_count() as usize;

        // Calculate stride for downsampling
        let stride = if element_count > max_elements {
            (element_count + max_elements - 1) / max_elements
        } else {
            1
        };

        let samples_to_read = (element_count + stride - 1) / stride;
        let samples_to_read = samples_to_read.min(max_elements);

        // Calculate byte range for the samples we need
        let end_offset = self.offset + (samples_to_read * stride) as u64 * self.bytes_per_element();

        // Ensure we don't go beyond the mapped region
        if end_offset as usize > mmap.len() {
            return Err("Tensor data extends beyond mapped region".to_string());
        }

        // Get the tensor data as bytes
        let tensor_data = &mmap[self.offset as usize..end_offset as usize];

        // Parse based on data type with proper dequantization
        match self.dtype {
            GGMLType::F32 => {
                let chunks = tensor_data.chunks_exact(4);
                Ok(chunks
                    .step_by(stride)
                    .take(samples_to_read)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect())
            }
            GGMLType::F16 => {
                Ok(dequantize::f16_to_f32(tensor_data, samples_to_read))
            }
            GGMLType::I8 => {
                Ok(tensor_data
                    .iter()
                    .step_by(stride)
                    .take(samples_to_read)
                    .map(|&b| b as i8 as f32)
                    .collect())
            }
            GGMLType::I16 => {
                let chunks = tensor_data.chunks_exact(2);
                Ok(chunks
                    .step_by(stride)
                    .take(samples_to_read)
                    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32)
                    .collect())
            }
            GGMLType::I32 => {
                let chunks = tensor_data.chunks_exact(4);
                Ok(chunks
                    .step_by(stride)
                    .take(samples_to_read)
                    .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32)
                    .collect())
            }
            GGMLType::Q4_0 => {
                Ok(dequantize::q4_0(tensor_data, samples_to_read))
            }
            GGMLType::Q4_1 => {
                Ok(dequantize::q4_1(tensor_data, samples_to_read))
            }
            GGMLType::Q8_0 => {
                Ok(dequantize::q8_0(tensor_data, samples_to_read))
            }
            // For other quantized types, use simplified dequantization
            _ => {
                Ok(dequantize::simplified(tensor_data, samples_to_read))
            }
        }
    }

    /// Read a 2D slice using memory mapping with bulk row reading
    fn read_slice_mmap(
        &self,
        mmap: &memmap2::Mmap,
        slice: &crate::tensor_slice::SliceSelection,
        max_elements: usize,
    ) -> Result<Vec<f32>, String> {
        let (slice_height, slice_width) = slice.slice_shape();
        let slice_elements = (slice_height * slice_width) as usize;

        // Calculate downsampling if needed
        let stride = if slice_elements > max_elements {
            (slice_elements + max_elements - 1) / max_elements
        } else {
            1
        };

        let samples_to_read = (slice_elements + stride - 1) / stride;
        let samples_to_read = samples_to_read.min(max_elements);

        // Use bulk row reading for better performance
        if slice_width > 100 && stride == 1 {
            self.read_slice_bulk_rows(mmap, slice, samples_to_read)
        } else {
            self.read_slice_element_by_element(mmap, slice, stride, samples_to_read)
        }
    }

    /// Bulk row reading - reads entire rows at once for maximum efficiency
    fn read_slice_bulk_rows(
        &self,
        mmap: &memmap2::Mmap,
        slice: &crate::tensor_slice::SliceSelection,
        samples_to_read: usize,
    ) -> Result<Vec<f32>, String> {
        let (slice_height, slice_width) = slice.slice_shape();
        let mut result = Vec::with_capacity(samples_to_read);
        let bytes_per_element = self.bytes_per_element();

        // Calculate row stride for downsampling
        let row_stride = if slice_height as usize * slice_width as usize > samples_to_read {
            (slice_height as usize * slice_width as usize + samples_to_read - 1) / samples_to_read
        } else {
            1
        };

        let mut samples_read = 0;
        for row in (0..slice_height).step_by(row_stride.max(1)) {
            if samples_read >= samples_to_read {
                break;
            }

            // Calculate the contiguous range for this row
            let row_start_offset = slice.linear_offset(row, 0)
                .ok_or("Invalid slice offset")?;
            let row_end_offset = slice.linear_offset(row, slice_width - 1)
                .ok_or("Invalid slice offset")?;

            let start_byte = (self.offset + row_start_offset * bytes_per_element) as usize;
            let end_byte = (self.offset + row_end_offset * bytes_per_element + bytes_per_element) as usize;

            // Ensure we're within bounds
            if end_byte > mmap.len() {
                return Err("Slice data extends beyond mapped region".to_string());
            }

            // Read entire row at once
            let row_data = &mmap[start_byte..end_byte];

            // Parse row data based on type
            let row_values = self.parse_tensor_bytes(row_data, slice_width as usize, samples_to_read - samples_read)?;
            result.extend_from_slice(&row_values);
            samples_read += row_values.len();
        }

        Ok(result)
    }

    /// Element-by-element reading for downsampling or small slices
    fn read_slice_element_by_element(
        &self,
        mmap: &memmap2::Mmap,
        slice: &crate::tensor_slice::SliceSelection,
        stride: usize,
        samples_to_read: usize,
    ) -> Result<Vec<f32>, String> {
        let (slice_height, slice_width) = slice.slice_shape();
        let mut result = Vec::with_capacity(samples_to_read);
        let bytes_per_element = self.bytes_per_element();

        let mut samples_read = 0;
        for row in 0..slice_height {
            for col in 0..slice_width {
                // Apply downsampling
                let linear_idx = row * slice_width + col;
                if linear_idx as usize % stride != 0 {
                    continue;
                }

                if samples_read >= samples_to_read {
                    break;
                }

                // Get the linear offset in the full tensor
                let offset = slice.linear_offset(row, col)
                    .ok_or("Invalid slice offset")?;

                // Calculate byte position
                let start_byte = (self.offset + offset * bytes_per_element) as usize;
                let end_byte = start_byte + bytes_per_element as usize;

                // Ensure we're within bounds
                if end_byte > mmap.len() {
                    return Err("Slice element extends beyond mapped region".to_string());
                }

                // Read single element
                let element_data = &mmap[start_byte..end_byte];
                let value = self.parse_single_element(element_data)?;
                result.push(value);
                samples_read += 1;
            }

            if samples_read >= samples_to_read {
                break;
            }
        }

        Ok(result)
    }

    /// Parse tensor bytes based on data type
    fn parse_tensor_bytes(&self, data: &[u8], max_values: usize, limit: usize) -> Result<Vec<f32>, String> {
        let values_to_parse = max_values.min(limit);

        match self.dtype {
            GGMLType::F32 => {
                let chunks = data.chunks_exact(4);
                Ok(chunks
                    .take(values_to_parse)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect())
            }
            GGMLType::F16 => {
                Ok(dequantize::f16_to_f32(data, values_to_parse))
            }
            GGMLType::I8 => {
                Ok(data.iter()
                    .take(values_to_parse)
                    .map(|&b| b as i8 as f32)
                    .collect())
            }
            GGMLType::I16 => {
                let chunks = data.chunks_exact(2);
                Ok(chunks
                    .take(values_to_parse)
                    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32)
                    .collect())
            }
            GGMLType::I32 => {
                let chunks = data.chunks_exact(4);
                Ok(chunks
                    .take(values_to_parse)
                    .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32)
                    .collect())
            }
            GGMLType::Q4_0 => {
                Ok(dequantize::q4_0(data, values_to_parse))
            }
            GGMLType::Q4_1 => {
                Ok(dequantize::q4_1(data, values_to_parse))
            }
            GGMLType::Q8_0 => {
                Ok(dequantize::q8_0(data, values_to_parse))
            }
            _ => {
                Ok(dequantize::simplified(data, values_to_parse))
            }
        }
    }

    /// Parse a single element from bytes
    fn parse_single_element(&self, data: &[u8]) -> Result<f32, String> {
        match self.dtype {
            GGMLType::F32 => {
                if data.len() >= 4 {
                    Ok(f32::from_le_bytes([data[0], data[1], data[2], data[3]]))
                } else {
                    Err("Insufficient data for F32".to_string())
                }
            }
            GGMLType::F16 => {
                if data.len() >= 2 {
                    let chunks = data.chunks_exact(2);
                    let f16_vals: Vec<f32> = dequantize::f16_to_f32(data, 1);
                    Ok(f16_vals[0])
                } else {
                    Err("Insufficient data for F16".to_string())
                }
            }
            GGMLType::I8 => {
                if !data.is_empty() {
                    Ok(data[0] as i8 as f32)
                } else {
                    Err("Insufficient data for I8".to_string())
                }
            }
            GGMLType::I16 => {
                if data.len() >= 2 {
                    Ok(i16::from_le_bytes([data[0], data[1]]) as f32)
                } else {
                    Err("Insufficient data for I16".to_string())
                }
            }
            GGMLType::I32 => {
                if data.len() >= 4 {
                    Ok(i32::from_le_bytes([data[0], data[1], data[2], data[3]]) as f32)
                } else {
                    Err("Insufficient data for I32".to_string())
                }
            }
            _ => {
                // For quantized types, simplified normalization
                if !data.is_empty() {
                    Ok((data[0] as f32 - 128.0) / 128.0)
                } else {
                    Err("Insufficient data for quantized type".to_string())
                }
            }
        }
    }

    /// Read a single element from the file at the current position
    fn read_single_element<R: Read>(&self, reader: &mut R) -> Result<f32, String> {
        match self.dtype {
            GGMLType::F32 => read_f32(reader),
            GGMLType::F16 => read_f16(reader),
            GGMLType::I8 => {
                let mut buf = [0u8; 1];
                reader.read_exact(&mut buf)
                    .map_err(|e| format!("Read failed: {}", e))?;
                Ok(buf[0] as i8 as f32)
            }
            GGMLType::I16 => {
                let mut buf = [0u8; 2];
                reader.read_exact(&mut buf)
                    .map_err(|e| format!("Read failed: {}", e))?;
                Ok(i16::from_le_bytes(buf) as f32)
            }
            GGMLType::I32 => {
                let val = read_i32(reader)?;
                Ok(val as f32)
            }
            // For quantized types, read raw byte and normalize
            _ => {
                let mut buf = [0u8; 1];
                reader.read_exact(&mut buf)
                    .map_err(|e| format!("Read failed: {}", e))?;
                Ok((buf[0] as f32 - 128.0) / 128.0)
            }
        }
    }

    /// Read tensor data from file with optional slicing
    /// Returns a downsampled vector of f32 values
    /// If slice_selection is None, reads the entire tensor (with downsampling)
    /// If slice_selection is Some, only reads the specified 2D slice (much faster for large tensors)
    pub fn read_tensor_data_with_slice(
        &self,
        file_path: &Path,
        max_elements: usize,
        slice_selection: Option<&crate::tensor_slice::SliceSelection>
    ) -> Result<Vec<f32>, String> {
        use std::fs::File;
        use std::io::{Seek, SeekFrom};

        let mut file = File::open(file_path)
            .map_err(|e| format!("Failed to open file: {}", e))?;

        // If we have a slice selection, use the optimized path
        if let Some(slice) = slice_selection {
            return self.read_slice_optimized(&mut file, slice, max_elements);
        }

        // Fall back to reading entire tensor
        self.read_entire_tensor(&mut file, max_elements)
    }

    /// Read only the specified 2D slice from a multi-dimensional tensor
    /// This is much more efficient than reading the entire tensor
    /// Optimized with buffered reading and smart row-wise access patterns
    fn read_slice_optimized<R: Read + Seek>(
        &self,
        file: &mut R,
        slice: &crate::tensor_slice::SliceSelection,
        max_elements: usize,
    ) -> Result<Vec<f32>, String> {
        let (slice_height, slice_width) = slice.slice_shape();
        let slice_elements = (slice_height * slice_width) as usize;

        // Calculate downsampling if needed
        let stride = if slice_elements > max_elements {
            (slice_elements + max_elements - 1) / max_elements
        } else {
            1
        };

        let samples_to_read = (slice_elements + stride - 1) / stride;
        let samples_to_read = samples_to_read.min(max_elements);
        let mut result = Vec::with_capacity(samples_to_read);

        let bytes_per_element = self.bytes_per_element();

        // For large slices, use row-wise buffered reading for better performance
        // This reduces the number of seeks dramatically (from N*M to N)
        if slice_width > 100 && stride == 1 {
            // Read entire rows at once (no downsampling, but large slice)
            let row_stride = if slice_height as usize > max_elements / slice_width as usize {
                (slice_height as usize * slice_width as usize + max_elements - 1) / max_elements
            } else {
                1
            };

            let mut samples_read = 0;
            for row in (0..slice_height).step_by(row_stride.max(1)) {
                if samples_read >= samples_to_read {
                    break;
                }

                // Calculate the contiguous range of elements for this row
                let row_start_offset = slice.linear_offset(row, 0)
                    .ok_or("Invalid slice offset")?;
                let row_size_bytes = (slice_width * bytes_per_element) as usize;

                // Seek to the start of the row
                let file_offset = self.offset + row_start_offset * bytes_per_element;
                file.seek(SeekFrom::Start(file_offset))
                    .map_err(|e| format!("Seek failed: {}", e))?;

                // Read the entire row into a buffer
                let mut row_buffer = vec![0u8; row_size_bytes];
                file.read_exact(&mut row_buffer)
                    .map_err(|e| format!("Read failed: {}", e))?;

                // Parse elements from the buffer
                let mut buf_reader = std::io::Cursor::new(row_buffer);
                for _ in 0..slice_width {
                    if samples_read >= samples_to_read {
                        break;
                    }

                    let value = self.read_single_element(&mut buf_reader)?;
                    result.push(value);
                    samples_read += 1;
                }
            }
        } else {
            // Original element-by-element reading with stride (for downsampling)
            let mut samples_read = 0;
            for row in 0..slice_height {
                for col in 0..slice_width {
                    // Apply downsampling
                    let linear_idx = row * slice_width + col;
                    if linear_idx as usize % stride != 0 {
                        continue;
                    }

                    if samples_read >= samples_to_read {
                        break;
                    }

                    // Get the linear offset in the full tensor
                    let offset = slice.linear_offset(row, col)
                        .ok_or("Invalid slice offset")?;

                    // Seek to this element
                    let file_offset = self.offset + offset * bytes_per_element;
                    file.seek(SeekFrom::Start(file_offset))
                        .map_err(|e| format!("Seek failed: {}", e))?;

                    // Read the value
                    let value = self.read_single_element(file)?;
                    result.push(value);
                    samples_read += 1;
                }

                if samples_read >= samples_to_read {
                    break;
                }
            }
        }

        Ok(result)
    }

    /// Read the entire tensor (existing behavior, now refactored)
    fn read_entire_tensor<R: Read + Seek>(
        &self,
        file: &mut R,
        max_elements: usize,
    ) -> Result<Vec<f32>, String> {
        file.seek(SeekFrom::Start(self.offset))
            .map_err(|e| format!("Failed to seek to tensor offset: {}", e))?;

        let element_count = self.element_count() as usize;

        // Calculate stride for downsampling
        let stride = if element_count > max_elements {
            (element_count + max_elements - 1) / max_elements
        } else {
            1
        };

        let samples_to_read = (element_count + stride - 1) / stride;
        let samples_to_read = samples_to_read.min(max_elements);

        let mut result = Vec::with_capacity(samples_to_read);

        // Read based on data type
        // For quantized types, we'll do a simplified read (just sample bytes)
        // For a production implementation, you'd want to properly dequantize
        match self.dtype {
            GGMLType::F32 => {
                for i in 0..samples_to_read {
                    let offset_in_tensor = (i * stride) as u64 * 4;
                    file.seek(SeekFrom::Start(self.offset + offset_in_tensor))
                        .map_err(|e| format!("Seek failed: {}", e))?;

                    let value = read_f32(file)?;
                    result.push(value);
                }
            }
            GGMLType::F16 => {
                for i in 0..samples_to_read {
                    let offset_in_tensor = (i * stride) as u64 * 2;
                    file.seek(SeekFrom::Start(self.offset + offset_in_tensor))
                        .map_err(|e| format!("Seek failed: {}", e))?;

                    let value = read_f16(file)?;
                    result.push(value);
                }
            }
            GGMLType::I8 => {
                for i in 0..samples_to_read {
                    let offset_in_tensor = (i * stride) as u64;
                    file.seek(SeekFrom::Start(self.offset + offset_in_tensor))
                        .map_err(|e| format!("Seek failed: {}", e))?;

                    let mut buf = [0u8; 1];
                    file.read_exact(&mut buf)
                        .map_err(|e| format!("Read failed: {}", e))?;
                    result.push(buf[0] as i8 as f32);
                }
            }
            GGMLType::I16 => {
                for i in 0..samples_to_read {
                    let offset_in_tensor = (i * stride) as u64 * 2;
                    file.seek(SeekFrom::Start(self.offset + offset_in_tensor))
                        .map_err(|e| format!("Seek failed: {}", e))?;

                    let mut buf = [0u8; 2];
                    file.read_exact(&mut buf)
                        .map_err(|e| format!("Read failed: {}", e))?;
                    result.push(i16::from_le_bytes(buf) as f32);
                }
            }
            GGMLType::I32 => {
                for i in 0..samples_to_read {
                    let offset_in_tensor = (i * stride) as u64 * 4;
                    file.seek(SeekFrom::Start(self.offset + offset_in_tensor))
                        .map_err(|e| format!("Seek failed: {}", e))?;

                    let value = read_i32(file)?;
                    result.push(value as f32);
                }
            }
            // For quantized types, read raw bytes and normalize to -1..1 range
            _ => {
                let bytes_per_element = (self.size_bytes as f64 / element_count as f64).ceil() as usize;
                for i in 0..samples_to_read {
                    let offset_in_tensor = (i * stride) as u64 * bytes_per_element as u64;
                    file.seek(SeekFrom::Start(self.offset + offset_in_tensor))
                        .map_err(|e| format!("Seek failed: {}", e))?;

                    let mut buf = [0u8; 1];
                    file.read_exact(&mut buf)
                        .map_err(|e| format!("Read failed: {}", e))?;
                    // Normalize byte value to -1..1 range
                    result.push((buf[0] as f32 - 128.0) / 128.0);
                }
            }
        }

        Ok(result)
    }

    /// Read tensor data from file, with optional quantization
    /// Returns a downsampled vector of f32 values
    /// This is the original method, maintained for backward compatibility
    pub fn read_tensor_data(&self, file_path: &Path, max_elements: usize) -> Result<Vec<f32>, String> {
        self.read_tensor_data_with_slice(file_path, max_elements, None)
    }
}

// Helper functions for reading binary data
fn read_u32<R: Read>(reader: &mut R) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)
        .map_err(|e| format!("Failed to read u32: {}", e))?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64, String> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)
        .map_err(|e| format!("Failed to read u64: {}", e))?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i32<R: Read>(reader: &mut R) -> Result<i32, String> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)
        .map_err(|e| format!("Failed to read i32: {}", e))?;
    Ok(i32::from_le_bytes(buf))
}

fn read_f32<R: Read>(reader: &mut R) -> Result<f32, String> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)
        .map_err(|e| format!("Failed to read f32: {}", e))?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f16<R: Read>(reader: &mut R) -> Result<f32, String> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf)
        .map_err(|e| format!("Failed to read f16: {}", e))?;
    let bits = u16::from_le_bytes(buf);

    // Convert f16 to f32
    // Simple conversion without external crate
    let sign = (bits >> 15) & 0x1;
    let exp = (bits >> 10) & 0x1f;
    let mantissa = bits & 0x3ff;

    let value = if exp == 0 {
        if mantissa == 0 {
            0.0
        } else {
            // Subnormal
            let mantissa_f = mantissa as f32 / 1024.0;
            mantissa_f * 2.0f32.powi(-14)
        }
    } else if exp == 0x1f {
        if mantissa == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        // Normalized
        let mantissa_f = 1.0 + (mantissa as f32 / 1024.0);
        mantissa_f * 2.0f32.powi(exp as i32 - 15)
    };

    Ok(if sign == 1 { -value } else { value })
}

fn read_string<R: Read>(reader: &mut R) -> Result<String, String> {
    let len = read_u64(reader)?;
    let mut buf = vec![0u8; len as usize];
    reader.read_exact(&mut buf)
        .map_err(|e| format!("Failed to read string: {}", e))?;
    String::from_utf8(buf)
        .map_err(|e| format!("Invalid UTF-8 string: {}", e))
}

fn read_metadata_kv<R: Read>(reader: &mut R) -> Result<(String, MetadataValue), String> {
    let key = read_string(reader)?;
    let value_type = read_u32(reader)?;

    let value = match value_type {
        0 => MetadataValue::UInt8(reader.read_exact(&mut [0u8; 1]).map(|_| 0).unwrap()),
        1 => MetadataValue::Int8(reader.read_exact(&mut [0u8; 1]).map(|_| 0).unwrap()),
        2 => MetadataValue::UInt16({
            let mut buf = [0u8; 2];
            reader.read_exact(&mut buf).unwrap();
            u16::from_le_bytes(buf)
        }),
        3 => MetadataValue::Int16({
            let mut buf = [0u8; 2];
            reader.read_exact(&mut buf).unwrap();
            i16::from_le_bytes(buf)
        }),
        4 => MetadataValue::UInt32(read_u32(reader)?),
        5 => MetadataValue::Int32(read_i32(reader)?),
        6 => MetadataValue::Float32(read_f32(reader)?),
        7 => MetadataValue::Bool(reader.read_exact(&mut [0u8; 1]).map(|_| true).unwrap()),
        8 => MetadataValue::String(read_string(reader)?),
        9 => {
            let arr_type = read_u32(reader)?;
            let arr_len = read_u64(reader)?;
            let mut arr = Vec::new();
            for _ in 0..arr_len {
                let elem = read_array_element(reader, arr_type)?;
                arr.push(elem);
            }
            MetadataValue::Array(arr)
        },
        10 => MetadataValue::UInt64(read_u64(reader)?),
        11 => {
            let mut buf = [0u8; 8];
            reader.read_exact(&mut buf).unwrap();
            MetadataValue::Int64(i64::from_le_bytes(buf))
        },
        12 => {
            let mut buf = [0u8; 8];
            reader.read_exact(&mut buf).unwrap();
            MetadataValue::Float64(f64::from_le_bytes(buf))
        },
        _ => return Err(format!("Unknown metadata type: {}", value_type)),
    };

    Ok((key, value))
}

fn read_array_element<R: Read>(reader: &mut R, elem_type: u32) -> Result<MetadataValue, String> {
    match elem_type {
        4 => Ok(MetadataValue::UInt32(read_u32(reader)?)),
        5 => Ok(MetadataValue::Int32(read_i32(reader)?)),
        6 => Ok(MetadataValue::Float32(read_f32(reader)?)),
        8 => Ok(MetadataValue::String(read_string(reader)?)),
        _ => Err(format!("Unsupported array element type: {}", elem_type)),
    }
}

fn read_tensor_info<R: Read>(reader: &mut R) -> Result<TensorInfo, String> {
    let name = read_string(reader)?;
    let n_dims = read_u32(reader)?;

    let mut dimensions = Vec::new();
    for _ in 0..n_dims {
        dimensions.push(read_u64(reader)?);
    }

    let dtype_value = read_u32(reader)?;
    let dtype = GGMLType::from_u32(dtype_value)?;

    let offset = read_u64(reader)?;

    // Calculate size based on dimensions and dtype
    let element_count: u64 = dimensions.iter().product();
    let bits_per_weight = dtype.bits_per_weight();
    let size_bytes = ((element_count as f64 * bits_per_weight as f64) / 8.0).ceil() as u64;

    Ok(TensorInfo {
        name,
        dimensions,
        dtype,
        offset,
        size_bytes,
    })
}

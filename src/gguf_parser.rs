use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in ASCII
const GGUF_VERSION: u32 = 3;

#[derive(Debug, Clone)]
pub struct GGUFFile {
    pub version: u32,
    pub metadata: GGUFMetadata,
    pub tensors: Vec<TensorInfo>,
    pub file_size: u64,
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
        })
    }

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

        let file = File::open(file_path)
            .map_err(|e| format!("Failed to open file: {}", e))?;

        // Use buffered reader for significantly faster I/O
        let mut buffered_file = BufReader::with_capacity(256 * 1024, file); // 256KB buffer

        // If we have a slice selection, use the optimized path
        if let Some(slice) = slice_selection {
            return self.read_slice_optimized(&mut buffered_file, slice, max_elements);
        }

        // Fall back to reading entire tensor
        self.read_entire_tensor(&mut buffered_file, max_elements)
    }

    /// Read only the specified 2D slice from a multi-dimensional tensor
    /// This is much more efficient than reading the entire tensor
    /// OPTIMIZED: Reads rows in chunks to minimize seeks
    fn read_slice_optimized<R: Read + Seek>(
        &self,
        file: &mut R,
        slice: &crate::tensor_slice::SliceSelection,
        max_elements: usize,
    ) -> Result<Vec<f32>, String> {
        let (slice_height, slice_width) = slice.slice_shape();
        let slice_elements = (slice_height * slice_width) as usize;

        // Calculate downsampling if needed
        let row_stride = if slice_elements > max_elements {
            let total_stride = (slice_elements + max_elements - 1) / max_elements;
            total_stride.max(1)
        } else {
            1
        };

        let col_stride = 1; // Read full rows, then downsample
        let rows_to_read = (slice_height as usize + row_stride - 1) / row_stride;
        let samples_per_row = slice_width as usize;

        let mut result = Vec::new();
        result.reserve(rows_to_read * samples_per_row / row_stride);

        let bytes_per_element = self.bytes_per_element();

        // Optimization: Read entire rows at once to minimize seeks
        // This reduces seeks from O(n*m) to O(n) for an nxm slice
        for row_idx in (0..slice_height as usize).step_by(row_stride) {
            let row = row_idx as u64;

            // Get offset to start of this row
            let row_start_offset = slice.linear_offset(row, 0)
                .ok_or("Invalid slice offset")?;
            let file_offset = self.offset + row_start_offset * bytes_per_element;

            // Seek once per row
            file.seek(SeekFrom::Start(file_offset))
                .map_err(|e| format!("Seek failed: {}", e))?;

            // Read entire row into buffer
            let row_bytes = (slice_width * bytes_per_element) as usize;
            let mut row_buffer = vec![0u8; row_bytes];
            file.read_exact(&mut row_buffer)
                .map_err(|e| format!("Read failed: {}", e))?;

            // Parse values from buffer (no more I/O here)
            let mut cursor = std::io::Cursor::new(&row_buffer);
            for col in (0..slice_width as usize).step_by(col_stride) {
                // Seek within the in-memory buffer (fast)
                cursor.set_position((col as u64) * bytes_per_element);
                let value = self.read_single_element(&mut cursor)?;
                result.push(value);
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

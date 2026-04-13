use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

// Magic number identifying our preprocessed format
const MAGIC: &[u8; 4] = b"MNIB";

pub struct MnistDataset {
    pub images: Vec<u8>,   // flat buffer: count * image_size bytes
    pub labels: Vec<u8>,   // count bytes, values 0–9
    pub image_size: usize, // bytes per image (784 raw, 98 after bit-packing)
}

impl MnistDataset {
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Returns the bit-packed pixel slice for sample `i`.
    pub fn image(&self, i: usize) -> &[u8] {
        &self.images[i * self.image_size..(i + 1) * self.image_size]
    }

    /// Returns a new dataset containing only the first `n` samples.
    pub fn subset(&self, n: usize) -> MnistDataset {
        let n = n.min(self.len());
        MnistDataset {
            images: self.images[..n * self.image_size].to_vec(),
            labels: self.labels[..n].to_vec(),
            image_size: self.image_size,
        }
    }

    /// Binarizes and bit-packs every image in-place.
    /// Pixels >= threshold → bit 1, below → bit 0.
    /// Each image shrinks from 784 bytes to 98 bytes (8 pixels per byte, MSB first).
    pub fn binarize(&mut self, threshold: u8) {
        let raw_size = self.image_size; // 784 before packing
        let packed_size = raw_size.div_ceil(8);
        let count = self.len();

        let mut packed = vec![0u8; count * packed_size];

        for i in 0..count {
            let src = &self.images[i * raw_size..(i + 1) * raw_size];
            let dst = &mut packed[i * packed_size..(i + 1) * packed_size];
            for (bit_idx, &pixel) in src.iter().enumerate() {
                if pixel >= threshold {
                    dst[bit_idx / 8] |= 1 << (7 - (bit_idx % 8));
                }
            }
        }

        self.images = packed;
        self.image_size = packed_size;
    }

    /// Saves the preprocessed dataset to a compact binary file.
    ///
    /// Format:
    ///   [4 bytes]  magic "MNIB"
    ///   [4 bytes]  sample count       (u32 little-endian)
    ///   [4 bytes]  image size (bytes) (u32 little-endian)
    ///   [count * image_size bytes]  all image data (flat)
    ///   [count bytes]               all labels
    pub fn save_preprocessed(&self, path: &Path) -> io::Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);

        writer.write_all(MAGIC)?;
        writer.write_all(&(self.len() as u32).to_le_bytes())?;
        writer.write_all(&(self.image_size as u32).to_le_bytes())?;
        writer.write_all(&self.images)?;
        writer.write_all(&self.labels)?;

        Ok(())
    }
}

/// Loads a preprocessed dataset saved with `save_preprocessed`.
pub fn load_preprocessed(path: &Path) -> io::Result<MnistDataset> {
    let mut reader = BufReader::new(File::open(path)?);

    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    assert_eq!(&magic, MAGIC, "not a preprocessed MNIST file");

    let mut buf4 = [0u8; 4];
    reader.read_exact(&mut buf4)?;
    let count = u32::from_le_bytes(buf4) as usize;

    reader.read_exact(&mut buf4)?;
    let image_size = u32::from_le_bytes(buf4) as usize;

    let mut images = vec![0u8; count * image_size];
    reader.read_exact(&mut images)?;

    let mut labels = vec![0u8; count];
    reader.read_exact(&mut labels)?;

    Ok(MnistDataset { images, labels, image_size })
}

/// Loads raw MNIST ubyte files.
pub fn load(images_path: &Path, labels_path: &Path) -> io::Result<MnistDataset> {
    let (images, image_size) = load_images(images_path)?;
    let labels = load_labels(labels_path)?;

    assert_eq!(
        labels.len(),
        images.len() / image_size,
        "image count != label count"
    );

    Ok(MnistDataset { images, labels, image_size })
}

fn load_images(path: &Path) -> io::Result<(Vec<u8>, usize)> {
    let mut reader = BufReader::new(File::open(path)?);

    let magic = read_u32_be(&mut reader)?;
    assert_eq!(magic, 2051, "unexpected magic number in images file: {magic}");

    let count      = read_u32_be(&mut reader)? as usize;
    let rows       = read_u32_be(&mut reader)? as usize;
    let cols       = read_u32_be(&mut reader)? as usize;
    let image_size = rows * cols;

    let mut images = vec![0u8; count * image_size];
    reader.read_exact(&mut images)?;

    Ok((images, image_size))
}

fn load_labels(path: &Path) -> io::Result<Vec<u8>> {
    let mut reader = BufReader::new(File::open(path)?);

    let magic = read_u32_be(&mut reader)?;
    assert_eq!(magic, 2049, "unexpected magic number in labels file: {magic}");

    let count = read_u32_be(&mut reader)? as usize;
    let mut labels = vec![0u8; count];
    reader.read_exact(&mut labels)?;

    Ok(labels)
}

fn read_u32_be(reader: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

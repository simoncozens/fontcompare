use ab_glyph_rasterizer::Rasterizer;
use harfbuzz_rs::{shape, DrawFuncs, Face, Font as HBFont, FontExtents, UnicodeBuffer};
use image::{imageops::crop_imm, GrayImage, ImageBuffer, Luma};
use kurbo::{BezPath, Rect, Shape};
use rayon::{iter::ParallelIterator, prelude::IntoParallelRefIterator};
use std::{
    cell::RefCell,
    collections::{BTreeMap, HashSet},
};
use thread_local::ThreadLocal;

use pyo3::prelude::*;

#[pyfunction]
fn diff_many_words(
    font_a: String,
    font_b: String,
    font_size: f32,
    wordlist: Vec<String>,
    threshold: f32,
) -> Vec<(String, String, String, f32)> {
    let results = _diff_many_words_parallel(&font_a, &font_b, font_size, wordlist, threshold);
    results
        .into_iter()
        .map(|d| (d.word, d.buffer_a, d.buffer_b, d.percent))
        .collect()
}

#[pymodule]
fn fontcompare(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(diff_many_words, m)?)?;
    Ok(())
}

#[derive(Debug)]
struct MyDrawFuncs {
    paths: Vec<BezPath>,
}

impl DrawFuncs for MyDrawFuncs {
    fn move_to(&mut self, _st: &harfbuzz_rs::draw_funcs::DrawState, to_x: f32, to_y: f32) {
        self.paths.push(BezPath::new());
        self.paths
            .last_mut()
            .unwrap()
            .move_to((to_x as f64, to_y as f64))
    }

    fn line_to(&mut self, _st: &harfbuzz_rs::draw_funcs::DrawState, to_x: f32, to_y: f32) {
        self.paths
            .last_mut()
            .unwrap()
            .line_to((to_x as f64, to_y as f64))
    }

    fn quadratic_to(
        &mut self,
        _st: &harfbuzz_rs::draw_funcs::DrawState,
        control_x: f32,
        control_y: f32,
        to_x: f32,
        to_y: f32,
    ) {
        self.paths.last_mut().unwrap().quad_to(
            (control_x as f64, control_y as f64),
            (to_x as f64, to_y as f64),
        )
    }

    fn cubic_to(
        &mut self,
        _st: &harfbuzz_rs::draw_funcs::DrawState,
        control1_x: f32,
        control1_y: f32,
        control2_x: f32,
        control2_y: f32,
        to_x: f32,
        to_y: f32,
    ) {
        self.paths.last_mut().unwrap().curve_to(
            (control1_x as f64, control1_y as f64),
            (control2_x as f64, control2_y as f64),
            (to_x as f64, to_y as f64),
        )
    }

    fn close_path(&mut self, _st: &harfbuzz_rs::draw_funcs::DrawState) {
        self.paths.last_mut().unwrap().close_path()
    }
}

fn count_differences(img_a: GrayImage, img_b: GrayImage) -> usize {
    let min_width = img_a.width().min(img_b.width());
    let min_height = img_a.height().min(img_b.height());
    let img_a = crop_imm(&img_a, 0, 0, min_width, min_height).to_image();
    let img_b = crop_imm(&img_b, 0, 0, min_width, min_height).to_image();
    let img_a_vec = img_a.to_vec();

    let differing_pixels = img_a_vec
        .iter()
        .zip(img_b.to_vec())
        .filter(|(&cha, chb)| ((cha) as i16 - *chb as i16).abs() != 0)
        .count();
    // let percent = differing_pixels as f32 / img_a_vec.len() as f32 * 100.0;
    differing_pixels
}

#[inline(always)]
fn p(p: kurbo::Point) -> ab_glyph_rasterizer::Point {
    ab_glyph_rasterizer::point(p.x as f32, p.y as f32)
}

struct PositionedGlyph {
    glyph_id: u32,
    x: f32,
    y: f32,
}

pub struct Renderer<'a> {
    hb_font: harfbuzz_rs::Owned<HBFont<'a>>,
    paths_cache: BTreeMap<u32, Vec<BezPath>>,
    extents: FontExtents,
}

impl Renderer<'_> {
    pub fn new(path: &str, font_size: f32) -> Self {
        let face = Face::from_file(path, 0).expect("No font");
        let mut hb_font = HBFont::new(face);
        hb_font.set_scale(font_size as i32, font_size as i32);
        let extents = hb_font.get_font_h_extents().unwrap();

        Self {
            hb_font,
            paths_cache: BTreeMap::new(),
            extents,
        }
    }

    #[inline(never)]
    fn glyphs_to_paths(&mut self, glyphs: Vec<PositionedGlyph>) -> Vec<BezPath> {
        let mut all_paths: Vec<BezPath> = vec![];
        for glyph in glyphs.into_iter() {
            let mut paths = self
                .paths_cache
                .entry(glyph.glyph_id)
                .or_insert_with(|| {
                    let drawer = MyDrawFuncs { paths: vec![] };
                    self.hb_font.draw_glyph(glyph.glyph_id, &drawer);
                    drawer.paths
                })
                .clone();
            // Translate by X,Y
            let translation = kurbo::Affine::translate((
                glyph.x as f64,
                glyph.y as f64 + self.extents.ascender as f64,
            ));
            for path in paths.iter_mut() {
                path.apply_affine(translation);
            }
            all_paths.extend(paths);
        }
        all_paths
    }

    fn rasterize(&self, all_paths: Vec<BezPath>, width: i32) -> Rasterizer {
        let mut rasterizer = Rasterizer::new(
            width as usize,
            (self.extents.ascender - self.extents.descender) as usize,
        );

        for path in all_paths.into_iter() {
            for seg in path.segments() {
                match seg {
                    kurbo::PathSeg::Line(l) => rasterizer.draw_line(p(l.p0), p(l.p1)),
                    kurbo::PathSeg::Quad(q) => rasterizer.draw_quad(p(q.p0), p(q.p1), p(q.p2)),
                    kurbo::PathSeg::Cubic(c) => {
                        rasterizer.draw_cubic(p(c.p0), p(c.p1), p(c.p2), p(c.p3))
                    }
                }
            }
        }
        rasterizer
    }

    fn to_image(&self, rasterizer: Rasterizer, width: i32) -> GrayImage {
        let dims = rasterizer.dimensions();
        let mut store = Vec::with_capacity(dims.0 * dims.1);
        rasterizer.for_each_pixel(|_, alpha| {
            let amount = (alpha * 255.0) as u8;
            store.push(amount);
        });
        // println!("Store length: {}", store.len());
        // println!("width: {}", width);
        // println!("height: {}", (extents.ascender - extents.descender));
        // println!(
        //     "expected: {}",
        //     width as usize * (extents.ascender - extents.descender) as usize
        // );
        // assert!(store.len() == width as usize * (extents.ascender - extents.descender) as usize);

        GrayImage::from_raw(
            width as u32,
            (self.extents.ascender - self.extents.descender) as u32,
            store,
        )
        .unwrap()
    }
    pub fn render_string(
        &mut self,
        string: &str,
    ) -> Option<(String, ImageBuffer<Luma<u8>, Vec<u8>>)> {
        let buffer = UnicodeBuffer::new().add_str(string);
        let output = shape(&self.hb_font, buffer, &[]);

        // The results of the shaping operation are stored in the `output` buffer.
        let positions = output.get_glyph_positions();
        let mut serialized_buffer = String::new();
        let infos = output.get_glyph_infos();
        let mut glyphs: Vec<PositionedGlyph> = vec![];
        let mut cursor = 0;
        let mut last_advance = 0;
        for (position, info) in positions.iter().zip(infos) {
            if info.codepoint == 0 {
                return None;
            }
            let x = (cursor + position.x_offset) as f32;
            let y = position.y_offset as f32;
            glyphs.push(PositionedGlyph {
                glyph_id: info.codepoint,
                x,
                y,
            });
            serialized_buffer.push_str(&format!("gid={},position={},{}|", info.codepoint, x, y));
            cursor += position.x_advance;
            last_advance = position.x_advance;
        }
        let last_width: i32 = infos
            .last()
            .and_then(|x| self.hb_font.get_glyph_extents(x.codepoint))
            .map(|x| x.width)
            .unwrap_or(0);

        let width = cursor - last_advance + last_width;

        let all_paths = self.glyphs_to_paths(glyphs);
        let rasterizer = self.rasterize(all_paths, width);
        let image = self.to_image(rasterizer, width);

        // Image will be drawn upside down, uncomment if you care.
        // flip_vertical_in_place(&mut image);

        serialized_buffer.pop();
        Some((serialized_buffer, image))
    }
}

#[derive(Debug)]
pub struct Difference {
    pub word: String,
    pub buffer_a: String,
    pub buffer_b: String,
    // pub diff_map: Vec<i16>,
    pub percent: f32,
}

fn _diff_many_words_parallel(
    font_a: &str,
    font_b: &str,
    font_size: f32,
    wordlist: Vec<String>,
    threshold: f32,
) -> Vec<Difference> {
    let tl_a = ThreadLocal::new();
    let tl_b = ThreadLocal::new();
    let tl_cache = ThreadLocal::new();
    let differences: Vec<Option<Difference>> = wordlist
        .par_iter()
        .map(|word| {
            let renderer_a = tl_a.get_or(|| RefCell::new(Renderer::new(font_a, font_size)));
            let renderer_b = tl_b.get_or(|| RefCell::new(Renderer::new(font_b, font_size)));
            let seen_glyphs: &RefCell<HashSet<String>> =
                tl_cache.get_or(|| RefCell::new(HashSet::new()));

            let (buffer_a, img_a) = renderer_a.borrow_mut().render_string(word)?;
            if buffer_a
                .split('|')
                .all(|glyph| seen_glyphs.borrow().contains(glyph))
            {
                return None;
            }
            for glyph in buffer_a.split('|') {
                seen_glyphs.borrow_mut().insert(glyph.to_string());
            }
            let (buffer_b, img_b) = renderer_b.borrow_mut().render_string(word)?;
            let percent = count_differences(img_a, img_b) as f32;

            Some(Difference {
                word: word.to_string(),
                buffer_a,
                buffer_b,
                // diff_map,
                percent,
            })
        })
        .collect();
    differences
        .into_iter()
        .flatten()
        .filter(|diff| diff.percent > threshold)
        .collect()
}

fn _diff_many_words_serial(
    font_a: &str,
    font_b: &str,
    font_size: f32,
    wordlist: Vec<String>,
    threshold: f32,
) -> Vec<Difference> {
    let mut renderer_a = Renderer::new(font_a, font_size);
    let mut renderer_b = Renderer::new(font_b, font_size);
    let mut seen_glyphs: HashSet<String> = HashSet::new();

    let mut differences: Vec<Difference> = vec![];
    for word in wordlist {
        let result_a = renderer_a.render_string(&word);
        if result_a.is_none() {
            continue;
        }
        let (buffer_a, img_a) = result_a.unwrap();
        if buffer_a.split('|').all(|glyph| seen_glyphs.contains(glyph)) {
            continue;
        }
        for glyph in buffer_a.split('|') {
            seen_glyphs.insert(glyph.to_string());
        }
        let result_b = renderer_b.render_string(&word);
        if result_b.is_none() {
            continue;
        }
        let (buffer_b, img_b) = result_b.unwrap();
        let percent = count_differences(img_a, img_b) as f32;
        if percent > threshold {
            differences.push(Difference {
                word: word.to_string(),
                buffer_a,
                buffer_b,
                // diff_map,
                percent,
            })
        }
    }
    differences
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::{BufRead, BufReader},
    };

    use super::*;

    #[test]
    fn test_it_works() {
        let file = File::open("test-data/Latin.txt").expect("no such file");
        let buf = BufReader::new(file);
        let wordlist = buf
            .lines()
            .map(|l| l.expect("Could not parse line"))
            .collect();
        use std::time::Instant;
        let now = Instant::now();

        let mut results = _diff_many_words_serial(
            "test-data/NotoSansArabic-Old.ttf",
            "test-data/NotoSansArabic-New.ttf",
            20.0,
            wordlist,
            10.0,
        );
        results.sort_by_key(|f| (f.percent * 100.0) as u32);
        // for res in results {
        //     println!("{}: {}%", res.word, res.percent)
        // }
        let elapsed = now.elapsed();
        println!("Elapsed: {:.2?}", elapsed);
    }

    #[test]
    fn test_render() {
        let mut renderer_a = Renderer::new("test-data/NotoSansArabic-Old.ttf", 20.0);
        let mut renderer_b = Renderer::new("test-data/NotoSansArabic-New.ttf", 20.0);
        let (_, image_a) = renderer_a.render_string("إلآ").unwrap();
        let (_, image_b) = renderer_b.render_string("إلآ").unwrap();
        let min_width = image_a.width().min(image_b.width());
        let min_height = image_a.height().min(image_b.height());
        let image_a = crop_imm(&image_a, 0, 0, min_width, min_height).to_image();
        let image_b = crop_imm(&image_b, 0, 0, min_width, min_height).to_image();
        image_a.save("image_a.png").expect("Can't save");
        image_b.save("image_b.png").expect("Can't save");
        let img_a_vec = image_a.to_vec();
        let img_b_vec = image_b.to_vec();
        let differing_pixels = img_a_vec
            .iter()
            .zip(img_b_vec)
            .filter(|(&cha, chb)| ((cha) as i16 - *chb as i16).abs() != 0)
            .count();
        let percent = differing_pixels as f32 / img_a_vec.len() as f32 * 100.0;
        println!("Percent: {:.2?}%", percent);
        assert!(percent < 3.0);
    }
}

fn main() {
    let results: Vec<(i32, i32)> = (1..3).flat_map(|i| (10..13).map(move |k| (i, k))).collect();
    println!("results: {:?}", results);
}

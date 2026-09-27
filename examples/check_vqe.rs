fn main() {
    let n_active_electrons = 2;
    let nbits = 6;

    let mut siter = Vec::new();
    let mut diter = Vec::new();

    for i in 0..n_active_electrons {
        for k in n_active_electrons..nbits {
            if (k + i) % 2 == 0 {
                siter.push((i, k));
            }
        }
    }

    for i in 0..n_active_electrons {
        for j in (i + 1)..n_active_electrons {
            for k in n_active_electrons..nbits {
                for l in (k + 1)..nbits {
                    if (i & 1) + (j & 1) == (k & 1) + (l & 1) {
                        diter.push((i, j, k, l));
                    }
                }
            }
        }
    }

    println!("siter: {:?}", siter);
    println!("diter: {:?}", diter);

    let n_occ = 2;
    let n_orb = 4;

    let oo_params_pairs: Vec<(usize, usize)> = (0..n_occ)
        .flat_map(|i| (n_occ..n_orb).map(move |k| (i, k)))
        .collect();
    let n_oo = oo_params_pairs.len();
    println!("oo_params: {:?}", oo_params_pairs);
    println!("n_oo: {}", n_oo);
}

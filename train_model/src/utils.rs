use std::fs;

pub fn read_dataset(name: &str) -> (Vec<u8>, Vec<u8>) {
    let data = fs::read_to_string(format!("dataset/{}", name)).unwrap();

    let mut target = Vec::new();
    let mut features = Vec::new();

    for line in data.lines() {
        let data = line
            .split(",")
            .map(|e| e.parse::<u8>().unwrap())
            .collect::<Vec<u8>>();

        target.push(data[0]);
        features.extend(&data[1..]);
    }

    (target, features)
}

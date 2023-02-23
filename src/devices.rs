#![allow(non_snake_case)]
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug)]
pub struct Response {
    requestDate: String,
    totalFeatures: usize,
    pub features: Vec<Info>,
}
#[derive(Deserialize, Serialize, Debug)]
pub struct Info {
    id: String,
    pub geometry: Geometry,
    pub properties: Properties,
}
#[derive(Deserialize, Serialize, Debug)]
pub struct Coordinates {
    pub lon: f64,
    pub lat: f64,
}
#[derive(Deserialize, Serialize, Debug)]
pub struct Geometry {
    r#type: String,
    coordinates: Coordinates,
    geometry_name: String,
}
#[derive(Deserialize, Serialize, Debug)]
pub struct Properties {
    pub device_name: String,
    pub active: bool,
    pub road_nl: String,
}

/// Get the devices from the bike counting api. If `request` is true, the data is retreived from
/// the file "devices.json". If `write` is `true`, the data
/// is written to "devices.json" as well.
pub async fn get_devices(
    request: bool,
    write: bool,
) -> Result<Response, Box<dyn std::error::Error>> {
    let devices = match request {
        false => serde_json::from_str::<Response>(
            &std::fs::read_to_string("./data/devices.json").unwrap(),
        )
        .unwrap(),
        true => {
            let client = reqwest::Client::new();
            let res = client
                .get("https://data.mobility.brussels/bike/api/counts/?request=devices")
                .send()
                .await?;
            serde_json::from_str::<Response>(&res.text().await?).unwrap()
        }
    };
    // Only bother writing if we have new info.
    if write && request {
        std::fs::write(
            "./data/devices.json",
            serde_json::to_string_pretty(&devices).unwrap(),
        )?;
    }
    Ok(devices)
}

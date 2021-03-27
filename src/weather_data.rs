#![allow(non_snake_case)]
use serde::{Deserialize, Serialize};
use std::vec::Vec;

#[derive(Deserialize, Serialize, Debug)]
struct Response {
    totalFeatures: usize,
    features: Vec<Feature>,
}
#[derive(Deserialize, Serialize, Debug)]
struct Feature {
    geometry: Geometry,
    properties: Data,
}
#[derive(Deserialize, Serialize, Debug)]
struct Geometry {
    coordinates: [f64; 2],
}
#[derive(Deserialize, Serialize, Debug)]
struct Data {
    code: usize,
    timestamp: String,
    air_pressure: usize,
    air_temperature: isize,
    relative_humidity: u8,
    precipitation: u8,
    wind_speed: u8,
}
#[derive(Deserialize, Serialize, Debug)]
pub struct AsosData {
    pub valid: String,
    pub tmpc: f64,
    pub sknt: f64,
}

fn calc_end_month(year: usize, month: u8) -> u8 {
    if month == 2 {
        return 28 + (year % 4 == 0) as u8;
    }
    31 - (month == 4 || month == 6 || month == 9 || month == 11) as u8
}
pub async fn fetch_from_month_and_year(
    month: u8,
    year: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // if month > 12 || month == 0 {
    //     return Err(Box::new("Month must be between 1 and 12."));
    // }
    let path = format!("./data/weather/hourly_{:02}_{}.csv", month, year);
    // if std::path::Path::new(&path).exists() && !update {
    //     return Ok(serde_json::from_str(
    //         &std::fs::read_to_string(path).unwrap(),
    //     )?);
    // }
    let start = format!("{}-{:02}-01", year, month);
    let end = format!("{}-{:02}-{:02}", year, month, calc_end_month(year, month));
    fetch_history(&path, &start, &end).await
}
pub async fn fetch_from_year(year: usize) -> Result<(), Box<dyn std::error::Error>> {
    // if month > 12 || month == 0 {
    //     return Err(Box::new("Month must be between 1 and 12."));
    // }
    let path = format!("./data/weather/hourly_{}.csv", year);
    // if std::path::Path::new(&path).exists() && !update {
    //     return Ok(serde_json::from_str(
    //         &std::fs::read_to_string(path).unwrap(),
    //     )?);
    // }
    let start = format!("{}-01-01", year);
    let end = format!("{}-12-31", year);
    fetch_history(&path, &start, &end).await
}

async fn fetch_history(
    path: &str,
    start: &str,
    end: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let base =
        "https://opendata.meteo.be/service/ows?service=WFS&version=2.0.0&request=GetFeature&";
    let main = "typenames=aws:aws_1hour&outputformat=json&CQL_FILTER=((BBOX(the_geom,3.201846,50.193663,5.255236,51.347375,%20%27EPSG:4326%27))";
    let url = format!("{}{}%20AND%20(timestamp%20%3E=%20%27{}%2000:00:00%27%20AND%20timestamp%20%3C=%20%27{}%2000:00:00%27))&sortby=timestamp",
        base, main,start, end);
    let response = reqwest::get(url).await?;
    println!("Received data from wep api");
    let data = serde_json::from_str::<self::Response>(&response.text().await?).unwrap();
    println!("Parsed {} data points", data.totalFeatures);
    let dir = "./data/weather/";
    if !std::path::Path::new(&dir).exists() {
        std::fs::create_dir_all(dir)?;
    }
    let mut writer = csv::WriterBuilder::new()
        .quote_style(csv::QuoteStyle::NonNumeric)
        .from_path(&path)?;
    print!("Writing data to {}: ", &path);
    for data in &data.features {
        writer.serialize(&data.properties)?;
    }
    println!("Done");
    writer.flush()?;
    Ok(())
}

#![allow(non_snake_case)]
use chrono::{Datelike, NaiveDate};
use serde::{Deserialize, Serialize};
use std::vec::Vec;

#[derive(Deserialize, Serialize, Debug)]
struct Response {
    requestDate: String,
    startDate: String,
    endDate: String,
    data: Vec<Data>,
}
#[derive(Deserialize, Serialize, Debug)]
struct Data {
    count_date: String,
    time_gap: u8,
    count: usize,
}
#[derive(Deserialize, Serialize, Debug)]
pub struct ParsedData {
    pub month: u32,
    pub weekday: String,
    pub date: String,
    pub hour: u8,
    pub count: usize,
}

impl ParsedData {
    fn from_data(data: Data) -> Self {
        let date = NaiveDate::parse_from_str(&data.count_date, "%Y/%m/%d").unwrap();
        ParsedData {
            month: date.month(),
            weekday: date.weekday().to_string(),
            date: date.to_string(),
            count: data.count,
            // Time gap is incremented every 15 mins, so 4 per hour
            hour: (data.time_gap - 1) / 4,
        }
    }
}

fn calc_end_month(year: usize, month: u8) -> u8 {
    if month == 2 {
        return 28 + (year % 4 == 0) as u8;
    }
    31 - (month == 4 || month == 6 || month == 9 || month == 11) as u8
}
pub async fn fetch_from_month_and_year(
    device_name: &str,
    month: u8,
    year: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // if month > 12 || month == 0 {
    //     return Err(Box::new("Month must be between 1 and 12."));
    // }
    let path = format!("./data/{device_name}/{month:02}_{year}.csv");
    // if std::path::Path::new(&path).exists() && !update {
    //     return Ok(serde_json::from_str(
    //         &std::fs::read_to_string(path).unwrap(),
    //     )?);
    // }
    let start = format!("{year}{month:02}01");
    let end = format!("{}{:02}{:02}", year, month, calc_end_month(year, month));
    fetch_history(device_name, &path, &start, &end).await
}
pub async fn fetch_from_year(
    device_name: &str,
    year: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // if month > 12 || month == 0 {
    //     return Err(Box::new("Month must be between 1 and 12."));
    // }
    let path = format!("./data/{device_name}/{year}.csv");
    // if std::path::Path::new(&path).exists() && !update {
    //     return Ok(serde_json::from_str(
    //         &std::fs::read_to_string(path).unwrap(),
    //     )?);
    // }
    let start = format!("{year}0101");
    let end = format!("{year}1231");
    fetch_history(device_name, &path, &start, &end).await
}

async fn fetch_history(
    device_name: &str,
    path: &str,
    start: &str,
    end: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Fetching device history from {device_name}");
    let url = format!(
        "https://data.mobility.brussels/bike/api/counts/?request=history&featureID={device_name}&startDate={start}&endDate={end}");
    let response = reqwest::get(url).await?;
    println!("Received data from wep api");
    let data = serde_json::from_str::<self::Response>(&response.text().await?)
        .unwrap()
        .data;
    println!("Parsed Data");
    let dir = format!("./data/{device_name}/");
    if !std::path::Path::new(&dir).exists() {
        std::fs::create_dir_all(dir)?;
    }
    print!("Writing data");
    let mut writer = csv::WriterBuilder::new()
        .quote_style(csv::QuoteStyle::NonNumeric)
        .from_path(path)?;
    // let mut writer = csv::Writer::from_path(&path)?;
    for data in data {
        writer.serialize(ParsedData::from_data(data))?;
    }
    println!(": Finished writing data");
    writer.flush()?;
    Ok(())
}

#![allow(non_snake_case)]
use chrono::{Datelike, NaiveDate};
use serde::{Deserialize, Serialize};
#[derive(Deserialize, Serialize, Debug)]
pub struct AsosData {
    pub valid: String,
    pub tmpc: f64,
    pub sknt: f64,
}

fn calc_end_month(year: i32, month: u32) -> u32 {
    if month == 2 {
        return 28 + (year as u32 % 4 == 0) as u32;
    }
    31 - (month == 4 || month == 6 || month == 9 || month == 11) as u32
}
pub async fn fetch_from_month_and_year(
    month: u32,
    year: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("./data/weather/asos_{month:02}_{year}.csv");
    let start = NaiveDate::from_ymd_opt(year, month, 1).unwrap();
    let end = NaiveDate::from_ymd_opt(year, month, calc_end_month(year, month))
        .unwrap()
        .succ_opt()
        .unwrap();
    fetch_history(&path, &start, &end).await
}
pub async fn fetch_from_year(year: i32) -> Result<(), Box<dyn std::error::Error>> {
    // if month > 12 || month == 0 {
    //     return Err(Box::new("Month must be between 1 and 12."));
    // }
    let path = format!("./data/weather/asos_{year}.csv");
    // if std::path::Path::new(&path).exists() && !update {
    //     return Ok(serde_json::from_str(
    //         &std::fs::read_to_string(path).unwrap(),
    //     )?);
    // }
    let start = NaiveDate::from_ymd_opt(year, 1, 1).unwrap();
    let end = NaiveDate::from_ymd_opt(year + 1, 1, 1).unwrap();
    fetch_history(&path, &start, &end).await
}

async fn fetch_history(
    path: &str,
    start: &NaiveDate,
    end: &NaiveDate,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "Getting weather history in period {start} to {end}"
    );
    let base =
        "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=EBBR&data=tmpc&data=sknt&";
    let tail = "&tz=Etc%2FUTC&format=onlycomma&latlon=no&elev=no&missing=M&trace=T&direct=no&report_type=1&report_type=2";
    // year1=2021&month1=1&day1=1&year2=2021&month2=4&day2=1
    let url = format!(
        "{}year1={}&month1={}&day1={}&year2={}&month2={}&day2={}{}",
        base,
        start.year(),
        start.month(),
        start.day(),
        end.year(),
        end.month(),
        end.day(),
        tail,
    );
    print!("Fetching data:");
    let response = reqwest::get(url).await?;
    println!(" Done");

    let dir = "./data/weather/";
    if !std::path::Path::new(&dir).exists() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, response.text().await?)?;
    println!("Finished writing data to {path}");
    // let data = serde_json::from_str::<self::Response>(&response.text().await?).unwrap();
    // println!("Parsed {} data points", data.totalFeatures);
    // let mut writer = csv::WriterBuilder::new()
    //     .quote_style(csv::QuoteStyle::NonNumeric)
    //     .from_path(&path)?;
    // print!("Writing data to {}: ", &path);
    // for data in &data.features {
    //     writer.serialize(&data.properties)?;
    // }
    // println!("Done");
    // writer.flush()?;
    Ok(())
}

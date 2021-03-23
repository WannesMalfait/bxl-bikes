#![allow(non_snake_case)]
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug)]
pub struct Response {
    requestDate: String,
    feature: String,
    pub data: Data,
}
#[derive(Deserialize, Serialize, Debug)]
pub struct Data {
    pub hour_cnt: usize,
    pub day_cnt: usize,
    pub year_cnt: usize,
    cnt_time: String,
}

/// Get the live data for the given device from the bike counting api.
pub async fn get_live_data(
    device_name: &str,
    client: &reqwest::Client,
) -> Result<self::Response, Box<dyn std::error::Error>> {
    let url = format!(
        "https://data.mobility.brussels/bike/api/counts/?request=live&featureID={}",
        &device_name
    );
    let response = client.get(url).send().await?;
    // println!("{}", response.text().await?);
    let data = serde_json::from_str::<self::Response>(&response.text().await?).unwrap();
    Ok(data)
}

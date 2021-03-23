mod device_data;
mod device_history;
mod devices;
mod weather_data;

async fn print_live_data() -> Result<(), Box<dyn std::error::Error>> {
    // Get all the available devices.
    let devices = devices::get_devices(false, false).await?;
    let client = reqwest::Client::new();
    for device in &devices.features {
        if !device.properties.active {
            continue;
        }
        let data = device_data::get_live_data(&device.properties.device_name, &client)
            .await?
            .data;
        println!(
            "{}:\nHour count: {}\nDay count:  {}\nYear count: {}",
            &device.properties.road_nl, data.hour_cnt, data.day_cnt, data.year_cnt
        );
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // print_live_data().await
    device_history::fetch_from_year("CB02411", 2020).await?;
    // weather_data::fetch_from_year(2021).await?;

    Ok(())
}

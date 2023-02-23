use chrono::{NaiveDateTime, Timelike};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use std::vec::Vec;
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
#[derive(Deserialize, Serialize, Debug)]
struct TrainingData {
    // Input to the model
    hour: u8,
    temperature: f64,
    windspeed: f64,
    weekday: u8,
    month: u32,
    year_count: usize, // Only up to the day before that.
    // Expected result from model
    day_count: usize,
}

fn make_training_data(device_name: &str, year: usize) -> std::io::Result<()> {
    println!(
        "Creating training data for {} from year {}",
        device_name, &year
    );
    let device_path = format!("./data/{}/{}.csv", device_name, year);
    if !std::path::Path::new(&device_path).exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Could not find data for {} from {}", device_name, &year),
        ));
    }
    let mut device_reader = csv::Reader::from_path(&device_path).unwrap();
    let mut device_iter = device_reader
        .deserialize::<device_history::ParsedData>()
        .peekable();
    let weather_path = format!("./data/weather/asos_{}.csv", &year);
    if !std::path::Path::new(&weather_path).exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Could not find data for weather of {}", year),
        ));
    }
    let mut weather_reader = csv::Reader::from_path(&weather_path).unwrap();
    let mut weather_iter = weather_reader
        .deserialize::<weather_data::AsosData>()
        .peekable();
    let mut day_count: usize = 0;
    let mut year_count: usize = 0;
    let mut training_data: Vec<TrainingData> = Vec::new();
    print!("Parsing data: ");
    let mut prev_d_data;
    if let Some(result) = device_iter.next() {
        prev_d_data = result?;
    } else {
        println!("No device data!");
        return Ok(());
    }
    let mut prev_w_data;
    if let Some(result) = weather_iter.next() {
        prev_w_data = result?;
    } else {
        println!("No weather data!");
        return Ok(());
    }
    loop {
        let weekday = chrono::Weekday::from_str(&prev_d_data.weekday).unwrap() as u8;
        let hour = prev_d_data.hour;
        let month = prev_d_data.month;
        let mut num_this_hour = prev_d_data.count;
        while let Some(result) = device_iter.next() {
            let d_data = result?;
            if d_data.hour != hour {
                day_count += num_this_hour;
                prev_d_data = d_data;
                break;
            }
            num_this_hour += d_data.count;
        }
        let mut temperature = 0.;
        let mut windspeed = 0.;
        let mut entries_this_hour = 0;
        let date = NaiveDateTime::parse_from_str(&prev_w_data.valid, "%Y-%m-%d %H:%M").unwrap();
        if date.hour() as u8 == hour {
            temperature += prev_w_data.tmpc;
            windspeed += prev_w_data.sknt;
            entries_this_hour += 1;
        } else {
            // Keep going till the correct hour.
            while let Some(result) = weather_iter.next() {
                let data = result?;
                let date = NaiveDateTime::parse_from_str(&data.valid, "%Y-%m-%d %H:%M").unwrap();
                if date.hour() as u8 == hour {
                    temperature += data.tmpc;
                    windspeed += data.sknt;
                    entries_this_hour += 1;
                    break;
                }
            }
        }
        while let Some(result) = weather_iter.next() {
            let data = result?;
            let date = NaiveDateTime::parse_from_str(&data.valid, "%Y-%m-%d %H:%M").unwrap();
            if date.hour() as u8 == hour {
                temperature += data.tmpc;
                windspeed += data.sknt;
                entries_this_hour += 1;
            } else {
                temperature = temperature / (entries_this_hour as f64);
                windspeed = windspeed / (entries_this_hour as f64);
                prev_w_data = data;
                break;
            }
        }
        training_data.push(TrainingData {
            hour,
            temperature,
            windspeed,
            weekday,
            month,
            year_count,
            day_count,
        });
        // A new day
        if prev_d_data.hour < hour {
            year_count += day_count;
            day_count = 0;
        }
        if device_iter.peek().is_none() || weather_iter.peek().is_none() {
            break;
        }
    }
    println!("Finished parsing data");
    let path = format!("./data/training_data/{}_{}.csv", device_name, &year);
    let dir = "./data/training_data/";
    if !std::path::Path::new(&dir).exists() {
        std::fs::create_dir_all(&dir)?;
    }
    let mut writer = csv::Writer::from_path(&path)?;
    print!("Writing data");
    for data in training_data {
        writer.serialize(data)?;
    }
    println!(": Finished writing data");
    writer.flush()?;
    return Ok(());
}

#[derive(Debug, Parser)]
#[command(author, version, about="Get data about bike counting poles in Brussels", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand, Clone)]
enum Commands {
    /// Get the live data from all available counting poles
    LiveData,
    /// Fetch device history and write it to disk
    DeviceHistory {
        year: usize,
        month: Option<u8>,
        /// Specify the name of the device to look up
        #[arg(default_value = "CBO2411", short, long)]
        device_name: String,
    },
    /// Fetch weather history for Brussels and write it to disk
    WeatherHistory { year: i32, month: Option<u32> },
    /// Create training data from the weather and device history of the given year
    TrainingData {
        year: usize,
        /// Specify the name of the device to look up
        #[arg(default_value = "CBO2411", short, long)]
        device_name: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Cli::parse();

    match args.command {
        Commands::LiveData => print_live_data().await?,
        Commands::DeviceHistory {
            year,
            month,
            device_name,
        } => {
            if let Some(month) = month {
                device_history::fetch_from_month_and_year(&device_name, month, year).await?
            } else {
                device_history::fetch_from_year(&device_name, year).await?
            }
        }
        Commands::WeatherHistory { year, month } => match month {
            Some(month) => weather_data::fetch_from_month_and_year(month, year).await?,
            None => weather_data::fetch_from_year(year).await?,
        },
        Commands::TrainingData { year, device_name } => make_training_data(&device_name, year)?,
    }

    Ok(())
}

# NHANES Hypertension Prediction Project

## Overview

This project analyzes NHANES (National Health and Nutrition Examination Survey) data to predict hypertension using various lifestyle factors. The pipeline consists of three main components:

1. **Data Combiner**: Combines and filters raw NHANES CSV files
2. **Data Autofiller**: Handles missing data and creates composite features
3. **Statistics Creator**: Generates comprehensive statistical analysis reports

## Features

### Data Processing Pipeline

- Combines multiple NHANES dataset files by year
- Filters relevant health and lifestyle variables
- Handles missing data through intelligent autofilling
- Creates composite health indicators
- Generates statistical reports and visualizations

### Key Health Indicators

- Blood Pressure Measurements
- Cholesterol Levels (Total, HDL, LDL)
- Smoking History
- Alcohol Consumption
- Diet and Nutrition
- Mental Health Factors
- Weight History
- Cardiovascular Health

## Project Structure

```bash
project/
├── data_combiner/
│   ├── core/
│   └── utils/
├── data_autofiller/
│   ├── core/
│   └── services/
├── statistics_creator/
│   ├── analyzers/
│   └── visualizers/
├── questions/
│   └── *.json
└── data/
    ├── raw/
    └── processed/
```

## Installation

```bash
# Install data_combiner dependencies
pip install -r data_combiner/requirements.txt

# Install statistics_creator dependencies
pip install -r statistics_creator/requirements.txt
```

## Usage

### 1. Data Combining

```python
from data_combiner import DataCombiner

# Combine and filter NHANES data files
combiner = DataCombiner(input_files)
combiner.combine_data()
```

### 2. Data Autofilling

```python
from data_autofiller.services import AutofillService

# Initialize service
service = AutofillService(
    data_reader=FileDataReader(),
    question_repository=FileQuestionRepository(),
    rule_engine=DefaultRuleEngine(),
    config=autofill_config
)

# Process files
service.process_files(input_files, output_dir)
```

### 3. Statistical Analysis

```python
from statistics_creator import StatisticsCreator

# Generate statistical reports
statistics_creator = StatisticsCreator(data_loader, analyzers, visualizers)
results = statistics_creator.run_analysis(data_path)
```

## Configuration

The project uses JSON configuration files in the questions/ directory to define:

- Required variables
- Valid value ranges
- Skip patterns
- Autofill rules
- Composite variable formulas

Example configuration:

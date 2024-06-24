# Project Title

## Description
This project involves training and applying an ABSA few-shot model and estimating a panel VAR model to analyze ESG aspects and sentiment in corporate reports.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/elenannat/ESG_ABSA.git
    cd ESG_ABSA
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Data
Due to size constraints, the data is not included in the repository. The datasets used in this project are:

- **Few-Shot Training and Test Data**: Available upon request.
- **ESG Scores**: Obtainable from providers.
- **Integrated Data**: Available upon request.

### Preprocessing
Preprocessing steps required for the data will be provided soon.

## Usage
To run the project, follow these steps:

1. **Prepare the Data**: Ensure all datasets are placed in the correct directories, e.g., `./Data/`. A sample structure will be provided soon.

2. **Train and Apply the ABSA Model**:
    ```bash
    python ABSA_training.py
    python ABSA_application.py
    ```

3. **Estimate the Panel VAR Model (R script)**:
    To estimate the Panel VAR model, you need to have R and the required packages installed. Open the `PanelVAR_ESG.R` script in R and run it.

## Models
### ABSA Few-Shot Model
This model extracts and analyzes ESG aspects and sentiment from sentences within corporate reports, providing a more nuanced analysis compared to traditional broad labels ('E', 'S', 'G').

#### Training Process
- Utilizes ~100 sentences from the datasets by Schimanski et al. (2024) for each ESG aspect.
- Sentences labeled by three experts.
- Model trained on NVIDIA RTX A5000 GPU.
- Default parameters used due to extensive training time.

#### Performance
- **Accuracy**: 91.73% (Entity), 79.17% (Sentiment)
- **F1 Score**: 91.84% (Entity), 79.99% (Sentiment)
- **Precision**: 92.04% (Entity), 80.96% (Sentiment)
- **Recall**: 91.73% (Entity), 79.17% (Sentiment)

### Panel VAR Model
Estimates panel VARs using ESG scores and net sentiment scores with fixed effects and System GMM approach. Data insufficiency prevented reliable subcategory level pVAR estimation.

## Results
- **Sentiment Analysis Results**: Stored in `./results_absa/`
- **Panel VAR Model Estimates**: Stored in `./Results/`

## Missing Files
Some files are not included in this repository due to size constraints or privacy issues. These files are:

- `./Data/SP_ESG/SPData.dta`
- `./Data/All_ESG/EuroStoxx50_2014-2024.dta`
- `./Data/All_ESG/EuroStoxx50_2014-2024_annual.dta`
- `./Data/All_ESG/EuroStoxx50-ticker-ciq-mapping.xlsx`
- `./Data/All_ESG/EuroStoxx50CompList.dta`
- `./Data/All_ESG/Refinitiv-ESG_Scores-2008-2024_ret2024-04-29.csv`

Please contact us to obtain these files.

## References
- Schimanski, T., Reding, A., Reding, N., Bingler, J., Kraus, M., & Leippold, M. (2024). Bridging the gap in ESG measurement: Using NLP to quantify environmental, social, and governance communication. *Finance Research Letters*, 61, 104979. doi:[10.1016/j.frl.2024.104979](https://doi.org/10.1016/j.frl.2024.104979).

## Contact
For any questions or support, please contact:
- **Name**: Elena Tönjes
- **Email**: elena.toenjes@web.de

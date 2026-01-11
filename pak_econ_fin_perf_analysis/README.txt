Purpose of academic project: Build a Pakistan Financial Stress Index, using data collected from 2005-2025 ideally (though past 10 years would be good too) for robust analysis.

Financial data needed: exchange rates, KSE-100 levels, interest spreads, interbank rates, reserve levels, non-performing loans (NPLs).
- Exchange rates: USDPKR spot data (daily or monthly) (Source: https://www.investing.com/currencies/usd-pkr-historical-data)
- Nominal effective exchange rate (daily or monthly) from https://easydata.sbp.org.pk/apex/f?p=10:211:7724461877834::NO:::
(NEER data is always end of month)
- Exchange market pressure (this is a theoretical measure which would need to be computed in the following manner):
Understanding Exchange Market Pressure
The EMP index is a theoretical measure, generally defined as the sum of two components, which are weighted to reflect the degree of central bank intervention in a managed exchange rate system: 
Change in the nominal exchange rate (depreciation/appreciation)
Change in foreign exchange reserves (which reflects the central bank's intervention to stabilize the currency) 
How to Obtain the Relevant Data
You can access the necessary data series from these sources:
State Bank of Pakistan (SBP): The SBP is the primary source for official, detailed data on Pakistan's economy. Their SBP EasyData portal or annual reports contain time-series data on:
Official Foreign Exchange Reserves: Available in USD millions.
Exchange Rates: Official Pak Rupee (PKR) to USD rates (interbank rates).
Domestic Credit/Money Supply: Measures like M0 or reserve money, which are often used in EMP models.
Interest Rates: Policy rates set by the SBP.
Financial Data Providers: Websites like Trading Economics provide readily accessible and chartable data, sourcing directly from the SBP and other official bodies. They offer:
Pakistan Foreign Exchange Reserves
USD/PKR Exchange Rate
Pakistan Interest Rate 
Calculating the EMP Index
Once you have obtained the time-series data, you would typically use an established methodology (such as the Girton and Roper (1977) or Weymark (1995) approaches) to calculate index
EXTENSION: Include EMP index computation
- KSE-100 index values (daily or monthly, ideally daily) (Source: https://www.investing.com/indices/karachi-100-historical-data) 
- Interest rate spread data (available from World Bank or SBP's general reports but not more granular frequency than monthly) (Source: https://easydata.sbp.org.pk/apex/f?p=10:211:16122404085233::NO:RP:P211_DATASET_TYPE_CODE,P211_PAGE_ID:TS_GP_BAM_SIRWALDR_M,1&cs=1A473715BC36293582D741D322134AC27)
- However, daily underlying data for interest rate spreads (interbank rates) (KIBOR rates) are published daily. Extension: Include this
- Pakistan interest rate (SBP benchmark policy rate data) (Source: https://tradingeconomics.com/pakistan/interest-rate): NB: SBP benchmark rates are not reported start of month but are usually reported end of month except for some cases. MPC dates don't follow a standard pattern.
- FX reserves (monthly data in https://tradingeconomics.com/pakistan/foreign-exchange-reserves)
- NPL data: Can't use this since we can only access quarterly data - a potential extension would be to get the quarterly data and blanket apply the quarterly values across all months in that quarter. Another extension would be to get the quarterly extension data from https://www.ceicdata.com/en/indicator/pakistan/non-performing-loans-ratio and use one of the following methods: step function or linear/cubic interpolation or mixed data sampling regression to estimate monthly values using just the observed quarterly data can be done. The 'EndDate' for eg if Nov-2025 represents the quarter from Sep-2025 to Nov-2025 inclusive.
 
Economic data needed: 
Output gap (real sector): This would need to be estimated using a macro-econometric model - can include this in future extension. 
trade finance data: Balance of trade monthly data from https://tradingeconomics.com/pakistan/balance-of-trade. 
remittances data: Data from https://easydata.sbp.org.pk/apex/f?p=10:211:9433821528928::NO:::  
inflation: monthly CPI data from https://tradingeconomics.com/pakistan/inflation-cpi

Political data needed: indicators of political instability.
- Terrorism index from https://tradingeconomics.com/pakistan/terrorism-index - might need to use step function or interpolation to construct estimated values at monthly level.

Purpose of the stress index: Offer early warnings for crises, by modelling it across time and finding patterns in it.
If we have monthly and daily data, we can construct a monthly variable out of daily data using an average value across days in a month for the daily variable, such that our dataset just has monthly data.

The 
We want 2 time series from the interbank spreads data:
1. Series Display Name = "Weighted Average Lending Deposit Rates  Lending Marginal (Overall)"
2. Series Display Name = "Weighted Average Lending Deposit Rates  Lending Marginal (Stocks)"
The differences between the "Overall" and the "Stocks" is timing of the loans being measured. "Overall"
measures the weighted average lending rate for new loans issued over a certain period such as a month for monthly data.
"Overall" reflects the current market conditions and immediate impact of recent monetary policy changes on new borrowers.
As a result, it is more volatile since it ignores existing loans in previous months/years. "Stocks" represents the weighted lending rate
for the entire portfolio of existing loans, and reflects actual interest income banks are earning from their loan book.
As a result, it is more stable and more sluggish to respond to monetary policy changes (unlike "Overall" which is weighted by new loan volume, it is weighted by total outstanding balance)
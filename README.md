# Reproducing the Figures in the Paper

This repository contains the raw aggregate data and the scripts needed to re-generate
every plot in the current paper.

```shell
$ python generate.py
```

The raw dataframes needed for each plot is available in `data` and the generated figures
are stored in the `plots` directory.

### Data format and description

| #         | Filename                                      | Columns                 | Description                                                                         |
|-----------|-----------------------------------------------|-------------------------|-------------------------------------------------------------------------------------|
| Figure 1  | figure-1-dataframe-asn-share.csv              | ASes                    | Number of Unique ASes Contacted                                                     |
|           |                                               | Count                   | Percentage of websites contacting the number of ASes                                |
|           | figure-1-dataframe-cdf.csv                    | cdf-X                   | Number of unique ASes Contacted                                                     |
|           |                                               | cdf-Y                   | Cumulative Sum of percentage of unique ASes Contacted                               |
| Figure 2  | figure-2.drawio                               |                         | Pre-rendered informational image of timeline reconstruction                         |
| Figure 3  | figure-3-dataframe.csv                        | dns                     | Number of DNS requests made during page load                                        |
|           |                                               | tls                     | Number of TLS requests made during page load                                        |
|           |                                               | ip_dns_tls              | Modelled expectation of DNS and TLS request counts in IP based coalescing           |
|           |                                               | origin_dns_tls          | Modelled expectation of DNS and TLS request counts in ORIGIN Frame based coalescing |
| Figure 4  | figure-4-dataframe.csv                        | original_san_list       | Number of DNS SAN Entries from certificates in measurement                          |
|           |                                               | updated_san_list        | Number of DNS SAN Entries from modelled ideal certificates                          |
| Figure 5  | figure-5-dataframe.csv                        | original_san_list       | Number of DNS SAN Entries from certificates in measurement                          |
|           |                                               | updated_san_list        | Number of DNS SAN Entries from modelled ideal certificates                          |
|           |                                               | mapped_san_additions    | Number of additions to make to original_san_list, each row is a certificate mapping |
| Figure 6  | figure-6.drawio                               |                         | Pre-rendered informational image about certificate changes during experiments       |
| Figure 7a | live_ip_coalescing_experiment_control.csv     | CDNJS-Coalesced         | Number of requests successfully coalescing cdnjs sub resources                      |
| Figure 7b | live_origin_coalescing_experiment_control.csv | CDNJS-Not-Coalesced     | Number of requests failed to coalesce cdnjs sub resources                           |
|           | Both the files have the same schema.          | Total-TLS               | Number of total TLS requests made                                                   |
|           |                                               | Type                    | Categorical variable - Experiment \| Control type                                   |
| Figure 8  | origin_frame_control_timeline.csv             | Time                    | Date format YYYY-MM-DD                                                              |
|           | origin_frame_experiment_timeline.csv          | Count                   | Number of TLS connections per second (100k)                                         |
| Figure 9a | figure-9a-dataframe.csv                       | page_load_times         | Measured page load time (ms)                                                        |
|           |                                               | ip_page_load_times      | Modelled IP based coalescing page load time (ms)                                    |
|           |                                               | as_page_load_times      | Modelled ORIGIN based coalescing page load time (ms)                                |
|           |                                               | cf_only_page_load_times | Modelled ORIGIN based coalescing if only Cloudflare deploys changes (ms)            |
| Figure 9b | figure-9b-dataframe.csv                       | Mode                    | Categorical variable - Experiment \| Control type                                   |
|           |                                               | Time                    | Measured page load time in active ORIGIN Frame experiments.                         |
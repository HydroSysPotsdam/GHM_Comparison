from pathlib import Path
import pandas as pd

AGGREGATED_PATH = Path("model_outputs/2b/aggregated/")
GNANN_PATH = Path("model_outputs/provided_by_gnann/")


if __name__ == "__main__":

    #### netrad_median
    
    netrad_median_paper = pd.read_csv(AGGREGATED_PATH.joinpath("netrad_median.csv"), index_col=False, header=0)
    netrad_median_gnann = pd.read_csv(GNANN_PATH.joinpath("netrad_median.csv"), index_col=False, header=0)

    netrad_median_paper.sort_values(["lat", "lon", "netrad"], inplace=True, ignore_index=True)
    netrad_median_gnann.sort_values(["lat", "lon", "netrad"], inplace=True, ignore_index=True)

    # print(netrad_median_paper)
    # print(netrad_median_gnann)

    pd.testing.assert_frame_equal(netrad_median_paper, netrad_median_gnann)

    #### netradiation clm45

    netradiation_clm45_paper = pd.read_csv(AGGREGATED_PATH.joinpath("clm45/netrad.csv"), index_col=False, header=0)
    netradiation_clm45_gnann = pd.read_csv(GNANN_PATH.joinpath("netradiation_clm45_minus.csv"), index_col=False, header=0)

    netradiation_clm45_paper.sort_values(["lat", "lon", "netrad"], inplace=True, ignore_index=True)
    netradiation_clm45_gnann.sort_values(["lat", "lon", "netrad"], inplace=True, ignore_index=True)

    # print(netradiation_clm45_paper)
    # print(netradiation_clm45_gnann)

    pd.testing.assert_frame_equal(netradiation_clm45_paper, netradiation_clm45_gnann)

    #### netradiation matsiro

    netradiation_matsiro_paper = pd.read_csv(AGGREGATED_PATH.joinpath("matsiro/netrad.csv"), index_col=False, header=0)
    netradiation_matsiro_gnann = pd.read_csv(GNANN_PATH.joinpath("netradiation_matsiro_withoutlakes.csv"), index_col=False, header=0)

    netradiation_matsiro_paper.sort_values(["lat", "lon", "netrad"], inplace=True, ignore_index=True)
    netradiation_matsiro_gnann.sort_values(["lat", "lon", "netrad"], inplace=True, ignore_index=True)

    # print(netradiation_matsiro_paper)
    # print(netradiation_matsiro_gnann)

    pd.testing.assert_frame_equal(netradiation_matsiro_paper, netradiation_matsiro_gnann)

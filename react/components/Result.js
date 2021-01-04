import React, { Component, Fragment } from "react";
import { connect } from "react-redux";
import PropTypes from "prop-types";

import { Bar } from "react-chartjs-2";

export class Result extends Component {
  static propTypes = {
    result: PropTypes.string.isRequired,
  };

  render() {
    const colors = [
      "206, 120, 255",
      "71, 160, 220",
      "218, 255, 0",
      "91, 200, 84",
      "255, 194, 193",
      "255, 66, 68",
      "255, 231, 151",
      "255, 167, 40",
      "242, 218, 254",
      "146, 101, 194",
      "220, 220, 170",
      "217, 129, 80",
    ];
    const data = {
      labels: this.props.classes,
      datasets: [
        {
          data: this.props.hyperplaneDistances,
          backgroundColor: this.props.hyperplaneDistances.map(
            (distance, index) => {
              if (distance > 0) {
                return "rgba(" + colors[index] + ", 1)";
              } else {
                return "rgba(" + colors[index] + ", 0.3)";
              }
            }
          ),
        },
      ],
    };
    const options = {
      title: {
        display: true,
        text: "Distance to separating hyperplane",
        fontColor: "white",
      },
      responsive: true,
      legend: {
        display: false,
      },
      scales: {
        xAxes: [
          {
            gridLines: {
              color: "rgba(255, 255, 255, 0.1)",
            },
            ticks: {
              fontColor: "white",
            },
            scaleLabel: {
              display: true,
              labelString: "Category",
              fontColor: "white",
            },
          },
        ],
        yAxes: [
          {
            gridLines: {
              color: "rgba(255, 255, 255, 0.1)",
            },
            ticks: {
              fontColor: "white",
            },
            scaleLabel: {
              display: true,
              labelString: "Distance",
              fontColor: "white",
            },
          },
        ],
      },
    };
    return (
      <Fragment>
        <h4>Predicted label: {this.props.predictedLabel}</h4>
        <Bar data={data} options={options} />
      </Fragment>
    );
  }
}

const mapStateToProps = (state) => ({
  result: state.results.result,
  predictedLabel: state.results.predictedLabel,
  hyperplaneDistances: state.results.hyperplaneDistances,
  classes: state.results.classes,
});

export default connect(mapStateToProps)(Result);

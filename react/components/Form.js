import React, { Component } from "react";
import { connect } from "react-redux";
import PropTypes from "prop-types";
import { getResults } from "../actions/results";

export class Form extends Component {
  state = {
    title: "",
    text: "",
  };

  static propTypes = {
    getResults: PropTypes.func.isRequired,
  };

  onChange = (e) => {
    e.preventDefault();
    this.setState({ [e.target.name]: e.target.value });

    var newState = this.state;
    newState[e.target.name] = e.target.value;

    this.props.getResults({
      title: newState.title,
      text: newState.text,
    });
  };

  componentDidMount() {
    this.props.getResults({
      title: this.state.title,
      text: this.state.text,
    });
  }

  render() {
    const { title, text } = this.state;
    return (
      <div className="card card-body mt-4 mb-4">
        <h2>Classify News</h2>
        <form>
          <div className="form-group">
            <label>Title</label>
            <input
              className="form-control"
              type="text"
              name="title"
              onChange={this.onChange}
              value={title}
            ></input>
          </div>
          <div className="form-group">
            <label>Text</label>
            <textarea
              className="form-control"
              type="text"
              name="text"
              onChange={this.onChange}
              value={text}
              rows="4"
            ></textarea>
          </div>
        </form>
      </div>
    );
  }
}

const mapStateToProps = (state) => ({
  title: state.results.title,
  text: state.results.text,
});

export default connect(mapStateToProps, { getResults })(Form);

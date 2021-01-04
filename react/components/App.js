import React, { Component, Fragment } from "react";
import ReactDOM from "react-dom";

import Header from "./Header";
import Form from "./Form";
import Result from "./Result";

import { Provider } from "react-redux";
import store from "../store";

class App extends Component {
  render() {
    return (
      <Provider store={store}>
        <Fragment>
          <Header />
          <div className="container">
            <Form />
            <Result />
          </div>
        </Fragment>
      </Provider>
    );
  }
}

ReactDOM.render(<App />, document.getElementById("app"));

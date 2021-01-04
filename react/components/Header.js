import React, { Component } from "react";

export class Header extends Component {
  render() {
    return (
      <nav className="navbar navbar-light bg-light">
        <div className="container-fluid">
          <a className="navbar-brand" href="http://huentelemu.com">
            huentelemu.com
          </a>
        </div>
      </nav>
    );
  }
}

export default Header;

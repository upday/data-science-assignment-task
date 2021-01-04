import axios from "axios";

import { GET_RESULTS } from "./types";

export const getResults = (texts) => (dispatch) => {
  console.log(texts);
  axios
    .post("upday-challenge/result", {
      title: texts.title,
      text: texts.text,
    })
    .then((res) => {
      dispatch({
        type: GET_RESULTS,
        payload: res.data,
        title: texts.title,
        text: texts.text,
      });
    })
    .catch((err) => console.log(err));
};

import { GET_RESULTS } from "../actions/types.js";

const initialState = {
  title: "",
  text: "",
  result: "",
  predictedLabel: "",
  hyperplaneDistances: [],
};

export default function (state = initialState, action) {
  switch (action.type) {
    case GET_RESULTS:
      return {
        ...state,
        result: action.payload.title_text,
        predictedLabel: action.payload.predicted_label,
        hyperplaneDistances: action.payload.hyperplane_distances,
        classes: action.payload.classes,
        title: action.title,
        text: action.text,
      };
    default:
      return state;
  }
}

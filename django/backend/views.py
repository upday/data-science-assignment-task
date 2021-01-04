from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .apps import BackendConfig

# Create your views here.
@csrf_exempt
def index(request):
    incoming_json = json.loads(request.body)
    title_text = f'{incoming_json["title"]} {incoming_json["text"]}'

    def lemmatize_text(text):
        new_text = ''
        for token in BackendConfig.nlp(text):
            lemma = token.lemma_
            if token.lemma_ != "-PRON-" and not token.is_punct and not token.is_stop:
                new_text += f" {lemma.lower()}"
        return new_text

    title_text = lemmatize_text(title_text)

    predicted_label = BackendConfig.predictor.predict([title_text])[0]
    hyperplane_distances = list(BackendConfig.predictor.decision_function([title_text])[0])
    classes = list(BackendConfig.predictor.classes_)
    return JsonResponse({
        'title_text': title_text,
        'predicted_label': predicted_label,
        'hyperplane_distances': hyperplane_distances,
        'classes': classes,
    })

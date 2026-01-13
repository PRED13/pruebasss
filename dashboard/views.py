import arff
import io
import pandas as pd
from django.shortcuts import render
from .utils import *

def index(request):
    if request.method == 'POST' and request.FILES.get('archivo_arff'):
        file = request.FILES['archivo_arff']
        # Leer ARFF desde memoria
        content = file.read().decode('utf-8')
        dataset = arff.loads(content)
        attributes = [attr[0] for attr in dataset['attributes']]
        df = pd.DataFrame(dataset['data'], columns=attributes)

        return render(request, 'dashboard/index.html', {
            'fotos_06': generate_visualizations_archivo_06(df),
            'fotos_07': generate_visualizations_archivo_07(df),
            'datos_08': generate_data_processing_08(df),
            'datos_05': generate_email_processing_05(df),
            'datos_09': generate_pipeline_processing_09(df),
            'titulo': f'Análisis: {file.name}',
            'inicial': False
        })
    
    return render(request, 'dashboard/index.html', {'inicial': True, 'titulo': 'Esperando archivo...'})
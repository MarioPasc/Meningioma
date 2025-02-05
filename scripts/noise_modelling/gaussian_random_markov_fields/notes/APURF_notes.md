# III Congreso Nacional de Estudiantes de Radiología y Medicina Física (APURF)

> Título propuesto: "Optimizando el Preprocesamiento de la Imagen por Resonancia Magnética: Estimación Ciega de Ruido con Campos Aleatorios Gaussianos"

## Introducción (Contexto)

Las redes neuronales convolucionales son el algoritmo de aprendizaje profundo preferido en los últimos años para llevar a cabo tareas de segmentación y clasificación de lesiones en neuroimagen. Estos algoritmos, si bien requieren de un ajuste de sus parámetros internos y externos, no podrían operar a tan alto rendimiento si no se les entrena con datos de calidad, que hayan pasado un proceso de curado cuidadoso.

Por otra parte, la resonancia magnética nuclear es una modalidad de imagen médica ampliamente utilizada en la mayoría de los entornos clínicos para la exploración de lesiones neuronales (e.g. tumores cerebrales, lesiones de ictus, etc.). Al igual que otras modalidades de imagen, este proceso no está exento de ruido en sus imágenes, cuya principal fuente de origen es el calor del paciente. Este ruido se convierte entonces en una característica intrínseca de la imagen, la cual puede ser estudiada y modelada mediante diversos modelos estadísticos.

Estudios han demostrado que el ruido de resonancia magnética es anisotrópico de manera global, es decir, que no parece hacer un patrón espacial de distribución del ruido uniforme o aleatorio, sino que se forman grumos de píxeles con intensidades correlacionadas. La distribución teórica de intensidades de píxeles de ruido de una resonancia magnética es una distribución de Rice, sin embargo, esto solo da la mitad de la información que se necesita para poder simular el ruido de fondo, que es la intensidad de los píxeles, aún se necesita una herramienta que pueda simular la distribución espacial de los píxeles.

Una etapa común en el preprocesamiento de las imágenes de resonancia magnética previa a la formación de un conjunto de datos de calidad es la estandarización de la resolución de la imagen. Dado que la recolección de datos para un conjunto de imágenes puede involucrar varios hospitales, y varias máquinas de resonancia magnética, estas imágenes pueden llegar con resoluciones muy distintas unas de otras, y se debe estandarizar a un tamaño concreto, primando la preservación de las características originales de la imagen. Estas características puede incluir: procentaje de cerebro que ocupa la imagen; relación de aspecto del cerebro en la imagen original; ruido de fondo de la imagen.

Una técnica común para preservar la relación de aspecto del cerebro mientras que se obtiene la resolución deseada es el letterboxing. Esta técnica consiste en aumentar de manera proporcional cada eje de la imagen hasta que uno de los dos obtiene la resolución deseada, mientras que el otro no llega. Se introducen entonces píxeles en el eje más corto hasta que se obtiene la resolución deseada en esa dirección (padding).

El problema que supone la aplicación de esta técnica es que, para mantener la concordancia de toda la imagen, se deben rellenar estos píxeles de manera que el ruido de fondo se expanda de manera contínua a estas nuevas secciones de la imagen. Este problema es relevante, ya que las redes neuronales convolucionales observan toda la imagen sin dar más o menos importancia a diferentes secciones de esta, por lo que cambios abruptos en el fondo de la imagen podrían introducir ruido o confusión en los parámetros internos de la red, que podrían desembocar en una bajada en el rendimiento de segmentación.

La mayoría de metodos de estimación fieles de ruido en resonancia magnética requieren de información que, en entornos clínicos, no se suele guardar con los datos del paciente, como el algoritmo de reconstrucción del k-espacio (e.g. GRAPPA, SENSE, etc.) o el método de fusión de la información de las diferentes bobinas para la formación de la imagen. Es por ello que se suele recurrir a los **métodos de estimación ciega**. Estos métodos recurren únicamente a la información encriptada en la imagen para aproximar una la función que define el ruido en la imagen determinada.

Muchos estudios que aplican aprendizaje profundo para la segmentación o clasificación de lesiones cerebrales ignoran el preprocesamiento de la imagen o no lo detallan en su metodología, dejando abiertas varias cuestiones relacionadas con las características de la imagen final, que podrían afectar al rendimiento al aplicas las tareas de visión por computador mediante aprendizaje profundo.

Este artículo presenta una nueva metodología para la estimación ciega del ruido de fondo de las resonancias magnética basada en la aplicación de campos aleatorios gaussianos. Este acercamiento pretende modelar las características no-estacionarias del ruido de resonancia magnética, además de la función de intensidad, de manera que se puedan generar cortes de ruido ilimitados a partir de estudios de resonancia magnética.

## Trabajos Futuros

Trabajos futuros incluyen el uso de esta herramienta para resolver el problema del rellenado de los píxeles en la técnica de letterboxing. Además, se pretende realizar el rellenado de estos píxeles utilizando diferentes técnicas (zero-padding, simulación del ruido con solo intensidad), y comprobar si existe una diferencia significativa en el rendimiento en la segmentación de lesiones, dependiente de la técnica que se use para replicar el ruido de fondo -o eliminarlo. 

## Referencias necesarias

1. Referencia que haya estudiado el ruido de fondo de resonancia magnetica y pruebe que es no estacionario
2. Referencias para las metodologías de submuestreo del k-espacio
3. Referencias para los métodos paramétricos que usan los datos adicionales que no se suelen tener para la emulación del ruido de fondo
4. Referencias a estudios de aprendizaje profundo que no usen reconstrucción del ruido de fondo o que usen técnicas que no preserven las características no estacionarias

## Tareas a realizar

- [ ] Aplicar la metodología a todo el conjunto de datos para obtener datos sobre los modelos de covarianza que mejor se ajustan y descubrir patrones sobre si, determinados modelos, se ajustan mejor a determinados pulsos.
- [ ] Encontrar una manera de validar el ruido original con el ruido generado (PSNR?)
- [ ] Entonces una manera de comprar la estructura espacial del ruido generado con la estructura del ruido original (Información Mutua local?)
- [ ] Escribir el resumen del artículo para darse de alta en el congreso
- [ ] Generar un abstract gráfico del pipeline generado
- [ ] Generar visualizaciones de cada paso

## Propuesta de resumen

Las redes neuronales convolucionales (CNN) requieren datos de calidad para segmentar lesiones cerebrales. La resonancia magnética nuclear (MRI), común en entornos clínicos, introduce ruido no estacionario de origen térmico, modelado teóricamente con distribuciones de Rice. Al estandarizar la resolución de imágenes de distinta procedencia, se añaden píxeles que generan discontinuidades en el fondo, lo cual altera los parámetros de la CNN, mermando su capacidad de delimitar lesiones. Sin información sobre la reconstrucción del k-espacio en muchos MRI de naturaleza clínica, los métodos de estimación únicamente basados en la imagen resultan esenciales para aproximar la distribución espacial e intensidad del ruido. Este artículo propone una metodología de estimación ciega de ruido en MRI basada en campos aleatorios gaussianos, que modela y genera ruido no estacionario, permitiendo la generación de cortes de ruido ilimitados para pre-procesamiento y aumentación de datos.

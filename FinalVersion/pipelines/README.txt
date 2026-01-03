Pipelines

For the 45 and 90 degree pipelines, they accept images
for patients facing to our left. To check for -45 and -90
degrees, the imput image needs to be simply mirrored (-45 -> 45)

In order to run a pipeline, open a terminal inside the folder
containing the pipeline python script and run:

python <NAME_OF_PIPELINE_SCRIPT.py> <INPUT_IMAGE>

some ready examples

    inside "/frontal": python FrontPipeline.py p12.jpg

    inside "/45_deg": python pipeline_45deg.py p022.jpg

    inside "/90_deg": python pipeline_90deg.py p002.jpg


It is possible to run all scripts individually by running:

    python <INDIVIDUAL_SCRIPT_NAME.py> <INPUT_IMAGE>

    or

    python <INDIVIDUAL_SCRIPT_NAME.py> 

Some scripts take the input image as argument, if not, they 
have the input image path hardcoded at the bottom of the
python file.


Please don't hesitate to contact me during any hour
of the day incase of any problems.

ataman01@ucy.ac.cy
99241183

Tamanis Andreas
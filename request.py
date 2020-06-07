from flask import request
# User defined utility functions
from utils import generate_random_start, generate_from_seed

# Home page
@app.route("/", methods=['GET', 'POST'])
def home():
    """Home page of app with form"""
    
    # Create form
    form = ReusableForm(request.form)

    # On form entry and all conditions met
    if request.method == 'POST' and form.validate():
        # Extract information
        seed = request.form['seed']
        diversity = float(request.form['diversity'])
        words = int(request.form['words'])
        # Generate a random sequence
        if seed == 'random':
            return render_template('random.html', 
                                   input=generate_random_start(model=model, 
                                                               graph=graph, 
                                                               new_words=words, 
                                                               diversity=diversity))
        # Generate starting from a seed sequence
        else:
            return render_template('seeded.html', 
                                   input=generate_from_seed(model=model, 
                                                            graph=graph, 
                                                            seed=seed, 
                                                            new_words=words, 
                                                            diversity=diversity))
    # Send template information to index.html
    return render_template('index.html', form=form)

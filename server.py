from flask import Flask, render_template
from flask import Flask, request, render_template, send_from_directory
import os
from flask import Flask, render_template, url_for, request, session, redirect
from flask_pymongo import PyMongo
import bcrypt


app = Flask(__name__)

app.config['MONGO_DBNAME'] = 'signup'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/signup'

mongo = PyMongo(app)
uploadfolder = 'D:/SIH 2019/Isro/static/Uploaded'
app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = uploadfolder
print("__name__is",__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
@app.route('/')
def index():
    return render_template("index.html")
@app.route('/dimg')
def dimg():
    return render_template("dimg.html")

@app.route('/dimg2')
def dimg2():
    return render_template("dimg2.html")

@app.route('/dimg3')
def dimg3():
    return render_template("dimg3.html")

@app.route('/dimg4')
def dimg4():
    return render_template("dimg4.html")

@app.route('/dimg5')
def dimg5():
    return render_template("dimg5.html")

@app.route('/dimg6')
def dimg6():
    return render_template("dimg6.html")

@app.route('/gmap')
def gmap():
    return render_template("gmap.html")
@app.route('/bhuvan')
def bhuvan():
    return render_template("bhuvan.html")






'''@app.route('/button/<filename>')
def send_image(filename):
    #print(foldername)
    return send_from_directory('bareily', filename)
@app.route('/button')
def button():
    name=request.args.get('para1')
    print('name',name)
    image_names=os.listdir('./bareily')
    return render_template("button.html",image_names=image_names)
    #return render_template("button.html")'''
#@app.route('/button')
#def buuton():
 #   return render_template("button.html")
@app.route('/login')
def message():
    if 'username' in session:
        return render_template('index.html')

    return render_template('login.html')
@app.route('/login', methods=['POST'])
def login():
    users = mongo.db.user
    login_user = users.find_one({'email' : request.form['username']})

    if login_user:
        if (request.form['pass'] == login_user['password']):
            session['username'] = request.form['username']
            session['logged_in'] = True
            return redirect(url_for('index'))

    return 'Invalid username/password combination'

@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        users = mongo.db.user
        existing_user = users.find_one({'email' : request.form['username']})

        if existing_user is None:
           # hashpass = bcrypt.hashpw(request.form['pass'].encode('utf-8'), bcrypt.gensalt())
            users.insert({'email' : request.form['username'], 'password' : request.form['pass']})
            session['username'] = request.form['username']
            return redirect(url_for('login'))

        return 'That username already exists!'

    return render_template('register.html')
@app.route('/logout')
def logout():
    session.pop('username',None)
    return redirect(url_for('login'))
#########################################################################################################################
@app.route('/upload')
def upload_files():
   return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save('uploaded.tif')
      os.system('move uploaded.tif static/uploaded.tif')
      os.chdir('static')
      os.system("python main.py 6 uploaded.tif")
      os.chdir('../')
      return render_template('up.html')


@app.route('/static1')
def button():
    print('Gondal')
    os.system("python main.py 9 Gondaloutput.tif")
    print('Done!!')
    return render_template("static1.html")
@app.route('/static2')
def button1():
    print('Rajkot')
    os.system("python ./static/main.py 9 Rajkotoutput.tif")
    print('Done!!')
    return render_template("static2.html")
@app.route('/static3')
def button2():
    print('Bareily')
    os.system("python ./static/main.py 9 Bareilyoutput.tif")
    print('Done!!')
    return render_template("static3.html")
@app.route('/static4')
def button3():
    print('Junagadh')
    os.system("python ./static/main.py 9 Junagadhoutput.tif")
    print('Done!!')
    return render_template("static4.html")
@app.route('/static5')
def button4():
    print('Moradabad')
    os.system("python ./static/main.py 9 Moradabadoutput.tif")
    print('Done!!')
    return render_template("static5.html")
@app.route('/static6')
def button5():
    print('Satna')
    os.system("python ./static/main.py 9 Satnaoutput.tif")
    print('Done!!')
    return render_template("static6.html")
    #return render_template("button.html")'''
'''@app.route('/gallery/<filename>')
def send_image(filename):
   # print(filename)
    return send_from_directory('bareily', filename)
@app.route('/gallery', methods=['POST'])
def bb():
        u =  request.form["xyz"]

        # url = "bb.html/para1=\"" + u + "\""
        image_names=os.listdir('./'+u)
        #print("in"+image_names)
        #print(u)
        return render_template("gallery.html",image_names=image_names)
        # print('my urk ',url)
        # return render_template("bb.html/para1=\"" + u + "\"")'''

if __name__=='__main__':
    app.secret_key = 'mysecret'
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.config.update(debug=True)
    app.run(debug=True)

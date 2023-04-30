import 'dart:convert';
import 'dart:io';

import 'package:firebase_auth/firebase_auth.dart' as auth;
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:flutter/cupertino.dart';
import 'package:gradproject/models/user.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:http/http.dart' as http;

class UserProvider extends ChangeNotifier {
  final auth.FirebaseAuth _auth = auth.FirebaseAuth.instance;
  final FirebaseStorage _storage = FirebaseStorage.instance;

  User? _user;
  final List<User> _users = [];

  User? get user => _user;
  List<User> get users => _users;

  signIn(String email, String password) async {
    final prefs = await SharedPreferences.getInstance();
    final credential = await _auth.signInWithEmailAndPassword(
        email: email, password: password);
    if (_users.isEmpty) {
      QuerySnapshot<Map<String, dynamic>> database =
          await FirebaseFirestore.instance.collection('users').get();

      List<Map<String, dynamic>> data =
          database.docs.map((doc) => doc.data()).toList();
      bool found = false;
      for (var item in data) {
        if (item['email'] == email) {
          _user = User(item['uid'], item['email'], item['username'],
              item['image'] ?? []);
          found = true;
        } else {
          _users.add(User(item['uid'], item['email'], item['username'],
              item['image'] ?? []));
        }
      }
      if (found) {
        await getImages();
        prefs.setString('email', email);
        prefs.setString('password', password);
        notifyListeners();
        return;
      }
    } else {
      for (var item in _users) {
        if (item.email == email) {
          _user = User(item.uid, item.email, item.username, item.image);
          await getImages();
          prefs.setString('email', email);
          prefs.setString('password', password);
          notifyListeners();
          return;
        }
      }
    }
  }

  createUser(String email, String password, String name) async {
    final prefs = await SharedPreferences.getInstance();
    final credential = await _auth.createUserWithEmailAndPassword(
        email: email, password: password);
    String image =
        'https://firebasestorage.googleapis.com/v0/b/mate-20088.appspot.com/o/no-img.png?alt=media&token=cdda0bcb-2b74-47f9-a1d5-591cca5ca625';
    await FirebaseFirestore.instance
        .collection('users')
        .doc(credential.user!.uid)
        .set({
      'username': name,
      'email': email,
      'image': image,
      'uid': credential.user!.uid,
    });
    QuerySnapshot<Map<String, dynamic>> database =
        await FirebaseFirestore.instance.collection('users').get();
    List<Map<String, dynamic>> data =
        database.docs.map((doc) => doc.data()).toList();
    bool found = false;
    for (var item in data) {
      User user = User(
          item['uid'], item['email'], item['username'], item['image'] ?? []);
      if (user.email != email) {
        _users.add(user);
      }
    }

    _user = User(credential.user!.uid, email, name, image);
    prefs.setString('email', email);
    prefs.setString('password', password);
    notifyListeners();
  }

  signOut() async {
    final prefs = await SharedPreferences.getInstance();
    await _auth.signOut();
    _user = null;
    _users.clear();
    prefs.remove('email');
    prefs.remove('password');
    notifyListeners();
  }

  Future<bool> uploadImage(String filePath, String fileName) async {
    File file = File(filePath);
    final name = user!.username;
    try {
      await _storage.ref('$name/$fileName').putFile(file);
      bool prediction = await detectFace(fileName);
      user!.patientsImages.clear();
      user!.patientsNames.clear();
      user!.patientsPredictions.clear();
      getImages();

      if (prediction) {
        return true;
      } else {
        return false;
      }
    } on FirebaseException catch (e) {
      print(e);
      return false;
    }
  }

  getImages() async {
    final name = user!.username;
    ListResult results = await _storage.ref('/$name').listAll();
    for (var ref in results.items) {
      String image = await ref.getDownloadURL();
      user!.addPatientName(ref.name);
      user!.addPatientImage(image);
      ListResult ress = await _storage
          .ref('/$name/' + ref.name.split('.')[0])
          .listAll();
      for (var reff in ress.items) {
        String image = await reff.getDownloadURL();
        user!.addPatientPrediction(image);
      }
    }
    print(user!.patientsPredictions);
  }

  upvoteModel(String name) async {
    QuerySnapshot<Map<String, dynamic>> database =
        await FirebaseFirestore.instance.collection('models').get();
    List<Map<String, dynamic>> data =
        database.docs.map((doc) => doc.data()).toList();
    for (var item in data) {
      if (item['name'] == name) {
        print(item);
        await FirebaseFirestore.instance
            .collection('models')
            .doc(item['uid'])
            .set({
          'name': item['name'],
          'upvotes': item['upvotes'] + 1,
          'uid': item['uid'],
        });
      }
    }
  }

  detectFace(String fileName) async {
    final name = user!.username;
    final ref = _storage.ref('$name/$fileName');
    String url = await ref.getDownloadURL();
    final res = await http.post(Uri.parse('http://10.0.2.2:5000/face'),
        headers: <String, String>{
          'Content-Type': 'application/json; charset=UTF-8',
        },
        body: jsonEncode(<String, String>{"url": url}));
    if (jsonDecode(res.body)['data'] == 'false') {
      await ref.delete();
      return false;
    } else {
      final r = await predictImage(url, fileName);
      if (r) {
        return true;
      } else {
        await ref.delete();
        return false;
      }
    }
  }

  predictImage(String url, String filename) async {
    final name = user!.username;
    final res = await http.post(Uri.parse('http://10.0.2.2:5000/predict'),
        headers: <String, String>{
          'Content-Type': 'application/json; charset=UTF-8',
        },
        body: jsonEncode(<String, String>{
          "url": url,
          "name": filename.split('.')[0],
          "doctor": name
        }));
    if (jsonDecode(res.body)['data'] == 'true') {
      return true;
    } else {
      return false;
    }
  }
}

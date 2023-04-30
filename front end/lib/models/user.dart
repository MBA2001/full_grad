import 'package:camera/camera.dart';

class User {
  String uid;
  String email;
  String username;
  String image;
  List<String> patientsImages = [];
  List<String> patientsNames= [];
  List<String> patientsPredictions = [];
  List<CameraDescription>? cameras;

  User(this.uid,this.email,this.username,this.image);

  addPatientImage(String patientImage){
    patientsImages.add(patientImage);
  }

  addPatientName(String patientName){
    patientsNames.add(patientName);
  }

  addPatientPrediction(String patientprediction){
    patientsPredictions.add(patientprediction);
  }
  
}

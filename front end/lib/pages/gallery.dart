import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:gradproject/models/user.dart';
import 'package:gradproject/pages/threed.dart';
import 'package:gradproject/providers/user_provider.dart';
import 'package:provider/provider.dart';
import 'package:url_launcher/url_launcher.dart';

class Gallery extends StatefulWidget {
  Gallery({Key? key}) : super(key: key);

  @override
  State<Gallery> createState() => _GalleryState();
}

class _GalleryState extends State<Gallery> {
  @override
  Widget build(BuildContext context) {
    final userProvider = Provider.of<UserProvider>(context);
    final user = Provider.of<UserProvider>(context).user!;
    List<String> patients = user.patientsImages;
    List<String> patientNames = user.patientsNames;
    List<String> patientsPredictions = user.patientsPredictions;
    return ListView.builder(
      padding: EdgeInsets.zero,
      scrollDirection: Axis.horizontal,
      itemCount: patients.length,
      itemBuilder: (BuildContext context, int index) {
        return Padding(
          padding: const EdgeInsets.fromLTRB(5, 0, 5, 4),
          child: Column(
            children: [
              IconButton(
                icon: Image.network(patients[index]),
                iconSize: 100,
                onPressed: () {
                  showDataAlert(patientsPredictions, index, userProvider, user,patientNames);
                  // Navigator.pushNamed(context, '/threed');
                },
              ),
              Text(patientNames[index]),
            ],
          ),
        );
      },
    );
  }

  openModel(List<String> patients, index, user) async {
    Uri url = Uri.parse(
        "http://192.168.1.36:8080/website/index.html#model=assets/" +
            patients[index].split('.')[0] +
            "_" +
            user.username +
            ".obj");
    if (!await launchUrl(url)) {
    throw Exception('Could not launch $url');
  }
  }

  showDataAlert(patients, int index, userProvider, User user,List<String> patientNames) {
    final models = ['CNN', 'Resnet', 'AlexNet'];
    showDialog(
        context: context,
        builder: (context) {
          return AlertDialog(
            shape: const RoundedRectangleBorder(
              borderRadius: BorderRadius.all(
                Radius.circular(
                  20.0,
                ),
              ),
            ),
            contentPadding: const EdgeInsets.only(
              top: 10.0,
            ),
            title: const Text(
              "Choose A model",
              style: TextStyle(fontSize: 24.0),
            ),
            content: Container(
              height: 400,
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(8.0),
                child: Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.start,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisSize: MainAxisSize.min,
                    children: <Widget>[
                      Row(
                        children: [
                          Padding(
                            padding: const EdgeInsets.fromLTRB(5, 0, 5, 4),
                            child: Column(
                              children: [
                                IconButton(
                                  icon: ClipRRect(
                                      borderRadius: BorderRadius.circular(20),
                                      child:
                                          Image.network(patients[index * 2])),
                                  iconSize: 100,
                                  onPressed: ()async{
                                    await openModel(patientNames,index,user);
                                  },
                                ),
                                Text(models[0]),
                              ],
                            ),
                          ),
                          Padding(
                            padding: const EdgeInsets.fromLTRB(5, 0, 5, 4),
                            child: Column(
                              children: [
                                IconButton(
                                  icon: ClipRRect(
                                      borderRadius: BorderRadius.circular(20),
                                      child: Image.network(
                                          patients[index * 2 + 1])),
                                  iconSize: 100,
                                  onPressed: ()async{
                                    await openModel(patientNames,index,user);
                                  },
                                ),
                                Text(models[1]),
                              ],
                            ),
                          ),
                        ],
                      ),
                      Padding(
                        padding: const EdgeInsets.fromLTRB(5, 0, 5, 4),
                        child: Column(
                          children: [
                            IconButton(
                              icon: ClipRRect(
                                  borderRadius: BorderRadius.circular(20),
                                  child: Image.network(patients[index * 2])),
                              iconSize: 100,
                              onPressed: ()async{
                                    await openModel(patientNames,index,user);
                                  },
                            ),
                            Text(models[2]),
                          ],
                        ),
                      ),
                      const SizedBox(
                        height: 20,
                      ),
                      const Text('choose the best model'),
                      Row(
                        children: [
                          ElevatedButton(
                            onPressed: () {
                              userProvider.upvoteModel('CNN');
                              Navigator.pop(context);
                            },
                            child: Text('CNN'),
                            style: ElevatedButton.styleFrom(
                              shape: CircleBorder(),
                              padding: EdgeInsets.all(20),
                              backgroundColor: Colors.white, // <-- Button color
                              foregroundColor: Colors.black, // <-- Splash color
                            ),
                          ),
                          ElevatedButton(
                            onPressed: () {
                              userProvider.upvoteModel('Resnet');
                              Navigator.pop(context);
                            },
                            child: Text('Resnet'),
                            style: ElevatedButton.styleFrom(
                              shape: CircleBorder(),
                              padding: EdgeInsets.all(20),
                              backgroundColor: Colors.white, // <-- Button color
                              foregroundColor: Colors.black, // <-- Splash color
                            ),
                          ),
                          ElevatedButton(
                            onPressed: () {
                              userProvider.upvoteModel('Alexnet');
                              Navigator.pop(context);
                            },
                            child: Text('AlexNet'),
                            style: ElevatedButton.styleFrom(
                              shape: CircleBorder(),
                              padding: EdgeInsets.all(20),
                              backgroundColor: Colors.white, // <-- Button color
                              foregroundColor: Colors.black, // <-- Splash color
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            ),
          );
        });
  }
}

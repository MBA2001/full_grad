// import 'package:final_project/modelss/user.dart';
// import 'package:final_project/providers/user_provider.dart';
import 'dart:io';
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:gradproject/providers/user_provider.dart';
import 'package:provider/provider.dart';
import 'package:loading_indicator/loading_indicator.dart';
import 'package:gradproject/widgets/textfield.dart';
// import 'package:profanity_filter/profanity_filter.dart';
// import 'package:provider/provider.dart';

class Home extends StatefulWidget {
  const Home({Key? key}) : super(key: key);

  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {
  String? path;
  String? extenstion;
  TextEditingController PatientNameController = TextEditingController();
  bool _validatePatientName = false;
  bool loading = false;
  @override
  Widget build(BuildContext context) {
    final keyboardOpen = MediaQuery.of(context).viewInsets.bottom != 0;
    final userProvider = Provider.of<UserProvider>(context);
    return Center(
      child: Column(
        children: [
          if (path != null && !keyboardOpen)
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: Image.file(
                File(path!),
                width: 200,
                height: 200,
              ),
            )
          else
            const SizedBox(
              height: 100,
            ),
          if(loading) Text('Please wait while',style: GoogleFonts.montserrat(textStyle:const  TextStyle(fontSize: 20))),
          if(loading) Text('the image is being uploaded',style: GoogleFonts.montserrat(textStyle:const  TextStyle(fontSize: 20))),
          if(loading) const SizedBox(height: 50,),
          if(!loading)
          SimpleTextField(
            Controller: PatientNameController,
            errorText: _validatePatientName ? 'The name cannot be empty' : null,
            hintText: 'Patient\'s Name',
            obscure: false,
          ),
          if(!loading)
          ElevatedButton(
            onPressed: () async {
              final results = await FilePicker.platform.pickFiles(
                allowMultiple: false,
                type: FileType.custom,
                allowedExtensions: ['png', 'jpg', 'jpeg'],
              );
              if (results == null) {
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(
                    content: Text('no file selected.'),
                  ),
                );
                return;
              } else {
                setState(() {
                  path = results.files.single.path;
                  extenstion = results.files.single.extension;
                });
              }
            },
            child: const Text('Choose your Image'),
          ),
          if(!loading)
          ElevatedButton(
            onPressed: () async {
              if (path == null || extenstion == null) {
                return;
              } else if (PatientNameController.text == '') {
                setState(() {
                  _validatePatientName = true;
                });
                return;
              } else {
                setState(() {
                  loading = true;
                });
                final filename = PatientNameController.text + '.' + extenstion!;
                print(filename);
                userProvider.uploadImage(path!, filename).then((value) {
                  if (value) {
                    setState(() {
                  loading = false;
                });
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(
                        content: Text('Image Uploaded successfully'),
                      ),
                      
                    );
                  } else {
                    setState(() {
                  loading = false;
                });
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(
                        content: Text('Please upload an Image of a face'),
                      ),
                    );
                  }
                });
              }
            },
            child: const Text('Upload Image'),
          ),
          if(loading)CircularProgressIndicator()
          
        ],
      ),
    );
  }
}

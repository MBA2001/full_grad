// import 'package:cloud_firestore/cloud_firestore.dart';
// import 'package:provider/provider.dart';
import 'package:provider/provider.dart';
import 'package:gradproject/providers/user_provider.dart';
import '../services/authservice.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import '../widgets/textfield.dart';
import 'package:google_fonts/google_fonts.dart';

class SignUp extends StatefulWidget {
  SignUp({Key? key}) : super(key: key);

  @override
  _SignUpState createState() => _SignUpState();
}

class _SignUpState extends State<SignUp> {
  bool _validate = false;
  bool _validateName = false;
  bool _validatePassword = false;
  bool _validateConfirmPass = false;
  String _passwordMessage = '';
  String emailMessage = '';
  final emailController = TextEditingController();
  final passwordController = TextEditingController();
  final usernameController = TextEditingController();
  final confirmPassController = TextEditingController();

  @override
  void dispose() {
    emailController.dispose();
    passwordController.dispose();
    super.dispose();
  }

  _signUpValidation(userProvider) async {
    if (passwordController.text.trim() == '' ||
        confirmPassController.text.trim() == '' ||
        usernameController.text.trim() == '' ||
        emailController.text.trim() == '') {
      setState(() {
        _passwordMessage = 'A field or more is empty, All fields are required';
        _validateConfirmPass = true;
        _validate = false;
        _validatePassword = false;
        _validateName = false;
      });
      return;
    } else if (passwordController.text != confirmPassController.text) {
      setState(() {
        _passwordMessage = 'confirm password must match password';
        _validateConfirmPass = true;
        _validate = false;
        _validatePassword = false;
        _validateName = false;
      });
      return;
    }
    try {
      QuerySnapshot<Map<String, dynamic>> database =
          await FirebaseFirestore.instance.collection('users').get();
      List<Map<String, dynamic>> data =
          database.docs.map((doc) => doc.data()).toList();
      for (var item in data) {
        if (item['username'] == usernameController.text) {
          setState(() {
            _validateName = true;
            _validateConfirmPass = false;
            _validate = false;
            _validatePassword = false;
          });
          return;
        }
      }
    } catch (e) {
      print(e);
      return;
    }
    setState(() {
      _validateConfirmPass = false;
    });

    try {
      await userProvider.createUser(emailController.text,
          passwordController.text, usernameController.text);
      Navigator.pop(context);
      Navigator.pop(context);

    } catch (e) {
      if (e.toString().contains('email address is already in use')) {
        setState(() {
          emailMessage = 'Email address is already in use';
          _validate = true;
          _validatePassword = false;
          _validateConfirmPass = false;
          _validateName = false;
        });
      } else if (e
          .toString()
          .contains('The email address is badly formatted')) {
        setState(() {
          emailMessage = 'Email address is badly formatted';
          _validate = true;
          _validatePassword = false;
          _validateConfirmPass = false;
          _validateName = false;
        });
      } else if (e
          .toString()
          .contains('Password should be at least 6 characters')) {
        setState(() {
          _validatePassword = true;
          _validate = false;
          _validateConfirmPass = false;
          _validateName = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final userProvider = Provider.of<UserProvider>(context);
    final keyboardOpen = MediaQuery.of(context).viewInsets.bottom != 0;

    return Scaffold(
      resizeToAvoidBottomInset: false,
      body: SafeArea(
        child: Column(
          children: [
            if (!keyboardOpen)
              const SizedBox(
                height: 20,
              ),
            if (!keyboardOpen)
              Image.asset(
                'assets/Cure.png',
                scale: 6,
              ),
            Text('Cure',
                style: GoogleFonts.sacramento(
                    textStyle: const TextStyle(fontSize: 40))),
            if (!keyboardOpen)
              const SizedBox(
                height: 10,
              ),
            SimpleTextField(
              Controller: usernameController,
              hintText: 'Name',
              errorText: _validateName ? 'Username already in use' : null,
              obscure: false,
            ),
            SimpleTextField(
              Controller: emailController,
              errorText: _validate ? emailMessage : null,
              hintText: 'Email',
              obscure: false,
            ),
            SimpleTextField(
              Controller: passwordController,
              errorText: _validatePassword
                  ? 'Password should be at least 6 characters'
                  : null,
              hintText: 'Password',
              obscure: true,
            ),
            SimpleTextField(
              Controller: confirmPassController,
              errorText: _validateConfirmPass ? _passwordMessage : null,
              hintText: 'Confirm Password',
              obscure: true,
            ),
            ElevatedButton.icon(
                onPressed: () async => (await _signUpValidation(userProvider)),
                icon: const Icon(Icons.app_registration),
                label: const Text('Register')),
          ],
        ),
      ),
    );
  }
}

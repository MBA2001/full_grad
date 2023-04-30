import 'package:flutter/material.dart';
import 'package:gradproject/providers/user_provider.dart';
import '../widgets/textfield.dart';
import 'package:provider/provider.dart';
import 'package:google_fonts/google_fonts.dart';
// import 'package:cloud_firestore/cloud_firestore.dart';




class LogIn extends StatefulWidget {
  LogIn({Key? key}) : super(key: key);

  @override
  _LogInState createState() => _LogInState();
}

class _LogInState extends State<LogIn> {
  TextEditingController emailController = TextEditingController();
  TextEditingController passwordController = TextEditingController();
  bool _validateEmail = false;
  bool _validatePass = false;
  String _validatePassText = '';

  @override
  void dispose() {
    emailController.dispose();
    passwordController.dispose();
    super.dispose();
  }

  _signInValidation(authservice) async {
    try {
      await authservice.signIn(emailController.text, passwordController.text);
      Navigator.pop(context);
    } catch (e) {
      print(e.toString());
      _validateEmail = false;
      _validatePass = false;
      if (e.toString().contains('address is badly formatted')) {
        setState(() {
          _validateEmail = true;
          _validatePass = false;
        });
      } else if (e.toString().contains(
              'no user record corresponding to this identifier. The user may have been deleted') ||
          e.toString().contains('The password is invalid')) {
        setState(() {
          _validatePass = true;
          _validateEmail = false;
          _validatePassText = 'Wrong email or password';
        });
      } else if (e.toString().contains('Given String is empty or null')) {
        setState(() {
          _validateEmail = false;
          _validatePass = true;
          _validatePassText = 'email and password cannot be empty';
        });
      } else {
        setState(() {
          _validateEmail = false;
          _validatePass = true;
          _validatePassText = 'something went wrong';
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
            const SizedBox(
              height: 20,
            ),
            if (!keyboardOpen)
              Image.asset(
                'assets/Cure.png',
                scale: 4,
              ),
              Text('Cure',style: GoogleFonts.sacramento(textStyle: TextStyle(fontSize: 40))),
            const SizedBox(
              height: 20,
            ),
            SimpleTextField(
              Controller: emailController,
              errorText: _validateEmail ? 'Please enter a valid email' : null,
              hintText: 'email',
              obscure: false,
            ),
            SimpleTextField(
              Controller: passwordController,
              errorText: _validatePass ? _validatePassText : null,
              hintText: 'password',
              obscure: true,
            ),
            ElevatedButton.icon(
                onPressed: () async => await _signInValidation(userProvider),
                icon: const Icon(Icons.login),
                label: const Text('login')),
            TextButton(
              onPressed: () {
                Navigator.pushNamed(context, '/register');
              },
              child: const Text('Register'),
            ),
          ],
        ),
      ),
    );
  }
}


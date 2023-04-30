import 'package:flutter/material.dart';


class SimpleTextField extends StatelessWidget {
  const SimpleTextField({
    Key? key,
    required this.Controller,
    String? errorText,
    required String hintText,
    required this.obscure,
  }) : _errorText = errorText,_hintText = hintText, super(key: key);

  final TextEditingController Controller;
  final String? _errorText;
  final String _hintText;
  final bool obscure;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.fromLTRB(7, 0, 7, 5),
      child: TextFormField(
        obscureText: obscure,
        controller: Controller,
        decoration: InputDecoration(
          hintText: _hintText,
          errorText: _errorText,
          border: OutlineInputBorder(
            borderRadius:  BorderRadius.circular(25.0),
            borderSide:const BorderSide(),
          ), 
        ),
      ),
    );
  }
}
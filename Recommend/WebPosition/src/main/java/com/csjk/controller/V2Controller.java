package com.csjk.controller;

import com.csjk.util.TfIdfUtil;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Arrays;
import java.util.List;

@RestController
@RequestMapping("/v2")
public class V2Controller {

        @CrossOrigin
        @GetMapping("/result")
        public List<String> get(String code){
            TfIdfUtil tfIdfUtil = new TfIdfUtil();
            String[] result = tfIdfUtil.handleTokens(code);
//            List<String> re = ;
            return Arrays.asList(result);
        }
}

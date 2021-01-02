# Notes for Web Development
+ Ref: Courera: https://www.coursera.org/learnhtml-css-javascript-for-web-developers/home/welcome
+ Setup
  + Chrome with CDT
  + Code Editor
  + Git
  + Browser Sync
    + need NodeJS
    + browser-sync start --server --directory -- files "*"
## HTML
+ Hypertext Markup Language
  + Define structure
  + Annotates content
+ History
  + W3C
    + HTML5(standard) 
  + WHATWG
    + HTML(evolving)
  + caniuse.com
  + browser statistics of w3c
+ HTML Tag
  + opening tag ... closing tag
    + ``` <p> content </p>```
    + no space allowed betweening the opening bracket and the foreword slash of the closing tag
  + only opening tag
    + ```<br>``` line break
    + ```<hr>```  horizontal rule
  + Attribute:name-value pair
    +  ```<p id="myId"></p>```
    + Attribute has to be unique within the scope of the entire HTML
    + can't specify attribute on the closing tag
    + at least one space betweeen the tag and any of its attributes, extra will be ignored
    + single or double quotes but must match
  + self-closing: no content
+ HTML Basis
  + from top to bottom
  + html/ex1
  + ```<!doctype html>```
    + render html
    + noncompliant-quircky mode  
  + ```<html> </html>```
    + ```<head> </head>```
      + ```<meta  charset utf-8>```
      + ```<title> </title>```
    + ```<body> </body>```
  + Content Model
    + Block-Level Element
      + begin on a new line(have its own line)
      + contain inline or other block
      + roughly flow contetn(HTML5 category)
      + ```<div></div>```
    + Inline Element 
      + same line
      + contain other inline
      + roughly pharsing content(HTML5 category)
      + ```<span></span>```
    + aligns with existing CSS rules
+ Essential HTML5 Tags
  + html/ex2
  + Heading Elements
    + in body,all block-level elements
    ```
    <h1>Main heading, crutial for search engine oriented<\h1>
    <h2>Subheading <\h2>    
    ```    
    + ```<header> header elemtn and navigation:<nav></nav>```
    ```
    <secion>
    <aside> Some information that relates to the main topic i.e. related posts</aside>
    </section>
    <footer> footer info</footer>
    ```
  + Lists
    + only li allowed in ul
    ```
    <ul>
    <li></li>
    <li><ul></ul></li>
    </ul> 
    ```
    + ```<ol> </ol>```
  + HTML Characketer Entity Refs
    + distinguish HTML character from content
    + escape < > &
      + use ```&lt; &gt; &amp;``` 
    + ```&copy;```
    + ```&nbsp;```
      + use between each word
      + misuse: for space;use span tag instead 
    + ```&quote;```  
  + Links
    + ```<a href="" title="" target="_blank"></a>```
    + ```<a href="" name="" target="_blank"></a>```
    + both inline and block
    + fragement specify ``` <a href="#section1"></a> ```
      then go to part ```<a name="section1"></a>```
  + Images
    + ```<img src="lena.jpg" width="400" height="360" alt="Picture with a quote">```
    + inline
    + width and height can help reserve space
    + when use url add ```<meta name="referrer" content="no-referrer">``` in head if can not dispaly(seems security issues)





## CSS
+ Style
  + color
  + Fontsize
+ CSS Rule
  + selector { #declaration
              property:value;}
    ``` 
    p{color:blue;
       font-size:20px;
       text-align:center;
    }
    ```
    + element selector: specify element type e.g. p
    + class selector:marked with class 

      ```
      .blue {
          color: blue;
      }
      <p class="blue"> </p>
      ```
  + 

  + id Selector: marked with id
  ```
  #name{
      color: blue;
  }
  ```

  + grouping selector

  ```
  div, .blue{
      color:blue;
  }
  ```
  + Combining Selectors
  + element with class: p with class
  
  ```
  p.big {
      font-size: 20px;
  }
  ```

  + Child Selector: p element as a direct child of article element

  ```
  article > p {
      color: blue;
  }
  ```

  + Descendant Selector:decendant not necessarily direct
  
  ```
  article p{
      color: blue;
  }
  '''

  + not limited to element selectors

  ```
  .colored p{
      color: blue;
  }

  p > .colored{
      color: blue'
  }
  ```

+ Pseudo-Class Selectors
  + selector:pseudo-class
  + predefined class names:
    + :link
    + :visited
    + :hover
    + :active: click but not release
    + :nth-child()
   
   ```CSS
   a:link, a:visited{
       text-decoration: none;
       border: 1px solid blue;
       display: block;
   }
   header li:nth-child(3){
       font-size: 24 px;
   }
   section div:nth-child(odd){
       background-color: gray;
   }
   ```
  
## Javascript
+ Behavior,Function

   

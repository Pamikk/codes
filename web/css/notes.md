# CSS Notes

+ Examples: css zen Garden
+ Anatomy of a CSS Rule: css-syntax.html
  + selector: p(all paragraphs)
  + property: color
  + value: blue

    ```CSS
    p {
        color:blue
        font-size:20 px;
    }
    ```

+ selectors
  + element selector: specify element type e.g. p
  + class selector:marked with class 

  ```CSS
  .blue {
      color: blue;
  }
  <p class="blue"> </p>
  ```

  + id Selector: marked with id
  ```CSS
  #name{
      color: blue;
  }
  ```

  + grouping slector

  ```CSS
  div, .blue{
      color:blue;
  }
  ```

+ Combining Selectors
  + element with class: p with class
  
  ```CSS
  p.big {
      font-size: 20px;
  }
  ```

  + Child Selector: p element as a direct child of article element

  ```CSS
  article > p {
      color: blue;
  }
  ```

  + Descendant Selector:decendant not necessarily direct
  
  ```CSS
  article p{
      color: blue;
  }
  '''

  + not limited to element selectors

  ```CSS
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
       text-dcoration: none;
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



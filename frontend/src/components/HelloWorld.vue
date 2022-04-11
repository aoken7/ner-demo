<template>
  <div>
    <button @click="getRandom">run ner</button><br>
    <textarea v-model="input_text" placeholder="add multiple lines"></textarea>
    <p> <span v-html="message"></span></p>
  </div>
</template>

<script>
import axios from 'axios'

const type_id_dict = {
    1: "人名",
    2: "法人名",
    3: "組織名",
    4: "組織名",
    5: "地名",
    6: "施設名",
    7: "製品名",
    8: "イベント名"
}

export default {
  data () {
    return {
      message: ""
    }
  },
  methods: {
    getRandom: async function () {
      this.message = await this.getRandomFromBackend()

      console.log(this.input_text)
      console.log(this.message)
      const obj = this.message['result'].reverse()
      var str = this.input_text
      //const str = '東京都は日本の首都です';
      obj.forEach(element => {
        const span = element['span']
        const id = element['type_id']
        const head = str.slice(0,span[0])
        const mid = str.slice(span[0],span[1])
        const end = str.slice(span[1])
        
        str = head + '<span class="ner type-'+id+'">' + mid
         + '</span><sub class="sub-'+id+'">'+type_id_dict[id]+'</sub> ' + end;
        this.message = str

      });

    },
    getRandomFromBackend: async function () {

      const path = 'http://150.89.233.81:5500/api/ner'
      await axios.post(path, {
        input_text: this.input_text
      })
      .then(response => {
        this.message = response.data
      })
      .catch(error => {
        console.log(error)
      })
      return this.message
  }
},
  created () {
    this.input_text = "  8日午前、日本のはるか南の海上で台風1号が発生しました。来週以降、日本に近づく可能性もあり、気象庁は今後の台風の情報に注意するよう呼びかけています。  気象庁の観測によりますと、8日午前9時、カロリン諸島で熱帯低気圧が台風1号に変わりました。  中心の気圧は1000ヘクトパスカル、中心付近の最大風速は18メートル、最大瞬間風速は25メートルで、中心から半径330キロ以内では風速15メートル以上の強い風が吹いています。  台風は1時間に15キロの速さで北西へ進んでいます。  台風は来週以降、日本に近づく可能性もあり、気象庁は今後の台風の情報に注意するよう呼びかけています。 "
  }
}
</script>

<style>
body {
  margin: 100px;
  font-size: 20px;
}
textarea {
  height: 12em;
  width: 35em;
  font-size: 16px;
}

button {
  padding: 5px;
  margin: 10px;
  font-size: 20px;
}

p {
  text-align: left;
}
.ner {
  border-style: solid;
  padding-top: 5px;
}
.type-1 {
  border-color: rgb(200, 200, 50);
}
.type-2 {
  border-color: rgb(200, 50, 50);
}
.type-3 {
  border-color: rgb(4, 0, 255);
}
.type-4 {
  border-color: rgb(4, 0, 255);
}
.type-5 {
  border-color: rgb(247, 142, 5);
}
.type-6 {
  border-color: rgb(204, 0, 255);
}
.type-7 {
  border-color: rgb(255, 0, 149);
}

.sub-1 {
  color: rgb(200, 200, 50);
}
.sub-2 {
  color: rgb(200, 50, 50);
}
.sub-3 {
  color: rgb(4, 0, 255);
}
.sub-4 {
  color: rgb(4, 0, 255);
}
.sub-5 {
  color: rgb(247, 142, 5);
}
.sub-6 {
  color: rgb(204, 0, 255);
}
.sub-7 {
  color: rgb(255, 0, 149);
}

</style>
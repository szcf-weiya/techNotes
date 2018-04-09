formStore={
    FILE_TYPE_LIST:FILE_TYPE_LIST,
    inject:function(injection){
        listeners=[],
        otherInfoPool={},
        currentState={
            page:0,
            data:{},
            valid:{},
            payment:{}
        },
        isMobile=injection.isMobile,
        getMode=injection.getMode,
        updateRawInfo=injection.updateRawInfo,
        setRawInfo=injection.setRawInfo,
        getContent=injection.getContent,
        getPostSettings=injection.getPostSettings
    }
    //...
}
currentState={
    page:0,
    data:{},
    valid:{},
    payment:{}
}



////////////////////////////////////////*
/*
setDefaultFillingData:function(){
    var isForce=arguments.length>0&&void 0!==arguments[0]&&arguments[0];
    if(1===getMode()){
        currentState.data||(currentState.data={});
        var d=_utils2["default"].getDefaultValue(), _currentAllComponent=[];
        do _currentAllComponent=_utils2["default"].getActive(),_lodash2["default"].forEach(_currentAllComponent,function(id){currentState.data.hasOwnProperty(""+id)&&currentState.data[id]&&!_lodash2["default"].isEmpty(currentState.data[id])&&!isForce||(currentState.data[id]=d[id])}),!currentState.payment.type&&_utils2["default"].processCart(currentState.data).length>0&&(currentState.payment=_lodash2["default"].assign({},currentState.payment,{type:_utils2["default"].getPaymentType()})),currentState=actionsProcess(currentState,{});while(_utils2["default"].getActive().length!==_currentAllComponent.length);_utils2["default"].setDefaultValue({})}},setUvdData:function(inWeChat){if(1===getMode()){currentState.data||(currentState.data={});var uvdInfo=_utils2["default"].getUvdContactInfo(),_currentAllComponent=[],_setList={};do _currentAllComponent=_utils2["default"].getActive(),_lodash2["default"].forEach(_currentAllComponent,function(id){var component=_utils2["default"].getComponents(id),_structureMapping=["I","II","III","IV"];if(uvdInfo[component.type]&&(!currentState.data[id]||_lodash2["default"].isEmpty(currentState.data[id])||inWeChat&&"wechat"===component.type)){var _ret6=function(){switch(component.type){case"gender":case"title":var item=_lodash2["default"].find(component.options,{id:+uvdInfo[component.type]});item&&(currentState.data[id]=[item.sId]);break;case"mobile":if(_setList[component.type])return{v:!0};_setList[component.type]=!0,component.isSendMsgAuth?currentState.data[id]={mobile:uvdInfo[component.type]||null}:currentState.data[id]=[uvdInfo[component.type]];break;case"email":case"website":case"tel":case"fax":case"wechat":case"qq":case"skype":case"whatsApp":case"messenger":case"viber":case"line":case"telegram":case"weibo":case"linkedIn":case"twitter":case"facebook":case"instagram":case"mkGooglePlus":if(_setList[component.type])return{v:!0};_setList[component.type]=!0,currentState.data[id]=[].concat(uvdInfo[component.type]);break;case"company":case"department":case"position":component.searchComponentType&&1!==+component.searchComponentType||(currentState.data[id]=uvdInfo[component.type]||currentState.data[id]);break;case"address":_structureMapping=_structureMapping.slice(0,+(component.minLevel||4));var _addressInfo=uvdInfo[component.type]||[{},null,null],_valid=!1;component&&component.isLimitCountry&&+component.limitCountry===+(_addressInfo[0]||{}).I&&0!==+component.limitCountry?_valid=!0:component&&!component.isLimitCountry&&0!==component.limitCountry&&(_valid=!0),_valid&&!function(){var _structure=[];_lodash2["default"].forEach(_structureMapping.slice(component.isLimitCountry?1:0),function(lvl){return!!_addressInfo[0][lvl]&&void _structure.push(_addressInfo[0][lvl])}),2===component.componentType?currentState.data[id]={struct:null,content:(currentState.data[id]||{}).content||_addressInfo[1]}:currentState.data[id]={struct:_lodash2["default"].isEmpty(_structure)?null:_structure,content:(currentState.data[id]||{}).content||_addressInfo[1],zipCode:(currentState.data[id]||{}).zipCode||_addressInfo[2]}}();break;case"location":_structureMapping=_structureMapping.slice(0,+(component.minLevel||3));var _locationInfo=uvdInfo[component.type]||{};component.isLimitCountry&&+component.limitCountry===+_locationInfo.I?_valid=!0:component.isLimitCountry||(_valid=!0),_valid&&!function(){var _structure=[];_lodash2["default"].forEach(_structureMapping.slice(component.isLimitCountry?1:0),function(lvl){return!!_locationInfo[lvl]&&void _structure.push(_locationInfo[lvl])}),2!==component.componentType&&(currentState.data[id]={struct:_lodash2["default"].isEmpty(_structure)?null:_structure})}();break;case"name":uvdInfo[component.type]&&(component.isOpenDetails?!function(){var legalNameInfo={};_lodash2["default"].isPlainObject(uvdInfo[component.type])&&_lodash2["default"].forEach(_FORM_COMPONENTS_NAME_DETAILS,function(info){(component.detailsItem||[2,4]).indexOf(+info.id)>-1&&uvdInfo[component.type][info.name]&&(legalNameInfo[info.name]=uvdInfo[component.type][info.name])}),_lodash2["default"].isEmpty(legalNameInfo)||(currentState.data[id]=legalNameInfo)}():!function(){var combineName=null;_lodash2["default"].isPlainObject(uvdInfo[component.type])?uvdInfo[component.type].n?combineName=uvdInfo[component.type].n:(combineName=[],_lodash2["default"].forEach(_FORM_COMPONENTS_NAME_DETAILS,function(info){uvdInfo[component.type][info.name]&&combineName.push(uvdInfo[component.type][info.name])}),combineName=combineName.join(" ")):uvdInfo[component.type]&&_lodash2["default"].isString(uvdInfo[component.type])&&(combineName=""+uvdInfo[component.type]),combineName&&(currentState.data[id]={n:combineName})}());break;default:uvdInfo[component.type]&&(currentState.data[id]=uvdInfo[component.type]||null)}}();if("object"===("undefined"==typeof _ret6?"undefined":_typeof(_ret6)))return _ret6.v}}),currentState.data=_lodash2["default"].assign({},currentState.data),currentState=_lodash2["default"].assign({},currentState),currentState=actionsProcess(currentState,{}),formStore.dispatch({type:"uvdUpdate"});while(_utils2["default"].getActive().length!==_currentAllComponent.length)}},setIpLoc:function(){if(1===getMode()){currentState.data||(currentState.data={});var ipLocInfo=_utils2["default"].getIpLoc();if(ipLocInfo&&!_lodash2["default"].isEmpty(ipLocInfo)){var _currentAllComponent=[];do _currentAllComponent=_utils2["default"].getActive(),_lodash2["default"].forEach(_currentAllComponent,function(id){var component=_utils2["default"].getComponents(id);if(("location"===component.type||"address"===component.type)&&(!currentState.data.hasOwnProperty(""+id)||!currentState.data[id]||_lodash2["default"].isEmpty(currentState.data[id]))){var _valid=!1;component.isLimitCountry&&+component.limitCountry===+ipLocInfo.I?_valid=!0:component.isLimitCountry||(_valid=!0),_valid&&2!==component.componentType&&(currentState.data[id]=_lodash2["default"].assign({},currentState.data[id]
*/
function getSubmitData(fillData){
    var _getComponentFillData=getComponentFillData(fillData,_logic),
    processedCtcfFillData=_getComponentFillData.processedCtcfFillData,
    processedFillData=_getComponentFillData.processedFillData,
    changedContactInfoList=_getComponentFillData.changedContactInfoList,
    content={cp:processedFillData};